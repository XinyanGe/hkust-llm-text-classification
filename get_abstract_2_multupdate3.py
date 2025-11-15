# -*- coding: utf-8 -*-
import os
import json
import time
import traceback
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from huggingface_hub import (
    HfApi,
    list_repo_tree,
    get_paths_info,
    repo_info,
    hf_hub_download,
    login,
)

# =======================
# 通用工具
# =======================
class ThreadSafeProgress:
    """优化的线程安全进度跟踪器"""
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_print = 0
    
    def update(self, success: bool = True):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            
            # 限制打印频率，避免 I/O 抖动
            now = time.time()
            if now - self.last_print >= 3 or self.completed % 50 == 0 or self.completed == self.total:
                elapsed = now - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                eta = (self.total - self.completed) / rate if rate > 0 else 0
                
                print(
                    f"📊 进度: {self.completed}/{self.total} "
                    f"({self.completed/self.total*100:.1f}%) | "
                    f"✅{self.successful} ❌{self.failed} | "
                    f"🚀{rate:.1f}/s | ETA: {eta/60:.1f}min"
                )
                self.last_print = now


class SmartRateLimiter:
    """智能令牌桶速率限制器"""
    def __init__(self, calls_per_second: float, burst_size: int = 10):
        self.rate = calls_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            # 补充令牌
            elapsed = now - self.last_refill
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                # 需要等待的时间
                wait_time = (1 - self.tokens) / self.rate
                return wait_time


def smart_rate_limit(calls_per_second: float, burst_size: int = 10):
    """智能速率限制装饰器"""
    limiter = SmartRateLimiter(calls_per_second, burst_size)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = limiter.acquire()
            if result is not True:  # 需要等待
                time.sleep(result)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =======================
# 主收集器
# =======================
class HighPerformanceSpaceCollector:
    """高性能 HuggingFace Spaces 数据收集器（修正版）"""

    def __init__(self, hf_token: str, output_dir: str = "./spaces_data",
                 max_workers: int = 20, file_workers: int = 8):
        self.hf_token = hf_token
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.file_workers = file_workers  # 单个 space 内文件处理并发数

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 日志（文件里尽量安静）
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(Path(output_dir) / "collection.log", encoding="utf-8")],
        )
        self.logger = logging.getLogger(__name__)

        # 线程本地存储
        self.local_data = threading.local()

        # 登录 Hugging Face
        try:
            if self.hf_token and self.hf_token.strip():
                login(token=self.hf_token)
            print("✅ HuggingFace 登录成功")
        except Exception as e:
            print(f"❌ HuggingFace 登录失败: {e}")
            raise

    # ---------- 内部工具 ----------
    def _hf_api(self) -> HfApi:
        if not hasattr(self.local_data, "hf_api"):
            self.local_data.hf_api = HfApi()
        return self.local_data.hf_api

    def _get_optimized_session(self):
        """若后续需要 HTTP 调用，可使用这个带连接池和重试的 session（当前元数据已改用 repo_info，不强依赖）"""
        if not hasattr(self.local_data, "session"):
            session = requests.Session()
            retry_strategy = Retry(
                total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy, pool_connections=20, pool_maxsize=50, pool_block=False
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update({
                "Authorization": f"Bearer {self.hf_token}",
                "User-Agent": "HighPerformanceSpaceCollector/2.0",
                "Connection": "keep-alive",
                "Accept-Encoding": "gzip, deflate",
            })
            self.local_data.session = session
        return self.local_data.session

    # ---------- 数据加载 ----------
    def load_spaces_from_csv(self, csv_path: str) -> List[str]:
        """优化的 CSV 加载：自动识别列名，否则取第一列"""
        try:
            df = pd.read_csv(csv_path, dtype=str, na_filter=False)
            possible_columns = ["space_name", "name", "spaces", "id", "space"]
            spaces = []
            for col in possible_columns:
                if col in df.columns:
                    spaces = df[col].tolist()
                    print(f"📊 从列 '{col}' 读取到 {len(spaces)} 个 spaces")
                    break
            if not spaces and len(df.columns) > 0:
                spaces = df.iloc[:, 0].tolist()
                print(f"📊 使用第一列读取到 {len(spaces)} 个 spaces")
            spaces = [s.strip() for s in spaces if s and str(s).strip()]
            return spaces
        except Exception as e:
            self.logger.error(f"❌ 加载 CSV 文件失败: {e}")
            return []

    def create_space_folder(self, space_name: str) -> str:
        """创建输出目录结构"""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in space_name)
        base = Path(self.output_dir) / safe
        for sub in ["app_files", "readme_files", "metadata", "other_files"]:
            (base / sub).mkdir(parents=True, exist_ok=True)
        return str(base)

    # ---------- 自动发现目标文件 ----------
    @smart_rate_limit(2.0, burst_size=8)
    def discover_target_files(self, space_name: str) -> List[str]:
        """
        枚举仓库文件，挑选常见入口/关注文件：
        - README.md / README.MD
        - app.py / app/app.py / src/app.py
        - main.py / src/main.py
        找不到时回退到 ["app.py", "README.md"]
        """
        try:
            tree = list_repo_tree(
                repo_id=space_name,
                repo_type="space",
                recursive=True,
                expand=False
            )
            candidates = set()
            for item in tree:
                p = getattr(item, "path", "") or ""
                low = p.lower()
                # 关心的文件
                if low.endswith("readme.md"):
                    candidates.add(p)
                elif low.endswith("/app.py") or low == "app.py":
                    candidates.add(p)
                elif low.endswith("/main.py") or low == "main.py":
                    candidates.add(p)
                # 若还有其他你想跟踪的文件，可按需补充
            # 提升可用性：优先 README，然后 app/main
            if not candidates:
                return ["app.py", "README.md"]
            # 排序：固定输出顺序便于稳定
            ordered = []
            for name in ["README.md", "Readme.md", "readme.md",
                         "app.py", "app/app.py", "src/app.py",
                         "main.py", "src/main.py"]:
                if name in candidates:
                    ordered.append(name)
            # 把非标准但匹配的也附上
            others = sorted([c for c in candidates if c not in ordered])
            return ordered + others
        except Exception:
            return ["app.py", "README.md"]

    # ---------- 历史与元数据 ----------
    @smart_rate_limit(1.5, burst_size=6)
    def _list_repo_commits(self, space_name: str):
        """列出空间仓库 commit 历史（倒序：新->旧）"""
        api = self._hf_api()
        return api.list_repo_commits(
            repo_id=space_name,
            repo_type="space",
            revision=None,      # 默认 main
            formatted=False
        )

    @smart_rate_limit(3.0, burst_size=10)
    def _file_changed_in_commit(self, space_name: str, filename: str, commit_id: str,
                                last_blob_id: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        检查某 commit 下指定文件的 blob_id 是否变化：
        - 文件存在，且 blob_id 与 last_blob_id 不同 => 变化
        - 文件不存在或相同 => 不变化
        """
        try:
            infos = get_paths_info(
                repo_id=space_name,
                paths=[filename],
                revision=commit_id,
                repo_type="space",
                expand=False
            )
            for info in infos:
                if getattr(info, "path", None) == filename:
                    blob = getattr(info, "blob_id", None)
                    if blob:
                        if last_blob_id is None or blob != last_blob_id:
                            return True, blob
                        return False, blob
            # 未找到该文件（此版本不存在）
            return False, last_blob_id
        except Exception:
            # 404 / 权限等异常：当作无变化，沿用 last_blob_id
            return False, last_blob_id

    @smart_rate_limit(2.0, burst_size=8)
    def get_file_commits(self, space_name: str, filename: str) -> List[Tuple[str, datetime]]:
        """
        正确的“文件级历史”实现：
        1) 取仓库 commit 列表（新->旧）
        2) 倒序遍历（旧->新）比较 blob_id，捕捉发生变化的提交
        返回 [(commit_id, created_at), ...]（旧->新）
        """
        try:
            commits = self._list_repo_commits(space_name)
        except Exception:
            return []

        # 倒序（旧 -> 新）比较
        commits_sorted = list(reversed(commits))
        results: List[Tuple[str, datetime]] = []
        last_blob_id: Optional[str] = None

        for c in commits_sorted:
            commit_id = getattr(c, "commit_id", None) or getattr(c, "oid", None)
            created_at = getattr(c, "created_at", None)
            # created_at 可能是 str，也可能是 datetime；统一成 datetime
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    created_at = datetime.now(timezone.utc)

            if not commit_id or not created_at:
                continue

            changed, last_blob_id = self._file_changed_in_commit(
                space_name, filename, commit_id, last_blob_id
            )
            if changed:
                results.append((commit_id, created_at))

        return results  # 已按旧->新

    def get_monthly_commits(self, commits: List[Tuple[str, datetime]]) -> List[Tuple[str, datetime]]:
        """按月抽取最近一次变更"""
        if not commits:
            return []
        monthly: Dict[Tuple[int, int], Tuple[str, datetime]] = {}
        for commit_id, cdate in commits:
            key = (cdate.year, cdate.month)
            hit = monthly.get(key)
            if (hit is None) or (cdate > hit[1]):
                monthly[key] = (commit_id, cdate)
        return sorted(monthly.values(), key=lambda x: x[1])

    @smart_rate_limit(2.0, burst_size=8)
    def get_space_metadata(self, space_name: str, commit_id: str) -> Optional[Dict]:
        """用文档化的 repo_info 获取指定 revision 的空间信息"""
        try:
            info = repo_info(
                repo_id=space_name,
                repo_type="space",
                revision=commit_id
            )
            return {
                "space_name": space_name,
                "commit_id": commit_id,
                "sdk": getattr(info, "sdk", None),
                "likes": getattr(info, "likes", 0),
                "created_at": getattr(info, "created_at", None),
                "updated_at": getattr(info, "last_modified", None),
                "tags": getattr(info, "tags", []) or [],
                "models": getattr(info, "models", []) or [],
                "datasets": getattr(info, "datasets", []) or [],
                "card_data": getattr(info, "card_data", {}) or {},
                "collection_time": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return None

    # ---------- 下载 ----------
    @smart_rate_limit(3.0, burst_size=10)
    def download_file_version(self, space_name: str, filename: str, commit_id: str,
                              commit_date: datetime, space_folder: str) -> Optional[Dict]:
        """下载特定 commit 的文件版本"""
        try:
            subfolder_map = {"app.py": "app_files", "README.md": "readme_files"}
            # 根据真实路径归类（若 path 中包含 app.py 或 main.py，也归到 app_files）
            low = filename.lower()
            if low.endswith("/app.py") or low.endswith("app.py") or low.endswith("main.py"):
                subfolder = "app_files"
            elif low.endswith("readme.md"):
                subfolder = "readme_files"
            else:
                subfolder = subfolder_map.get(filename, "other_files")

            local_dir = Path(space_folder) / subfolder
            local_dir.mkdir(parents=True, exist_ok=True)

            downloaded_path = hf_hub_download(
                repo_id=space_name,
                repo_type="space",
                revision=commit_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False
            )

            # 命名：用原始路径替换斜杠，避免冲突
            file_base = filename.replace("/", "__").replace(".", "_")
            date_str = commit_date.strftime("%Y%m%d")
            file_ext = Path(filename).suffix
            new_filename = f"{file_base}_{commit_id[:7]}_{date_str}{file_ext or ''}"
            new_path = local_dir / new_filename

            if Path(downloaded_path).exists():
                # 若下载的文件名与 new_path 不同则重命名
                if str(downloaded_path) != str(new_path):
                    Path(downloaded_path).rename(new_path)
                return {
                    "file_path": str(new_path),
                    "commit_id": commit_id,
                    "commit_date": commit_date.isoformat(),
                    "file_size": new_path.stat().st_size,
                    "source_file": filename
                }
        except Exception:
            pass
        return None

    # ---------- 单空间处理 ----------
    def process_file_parallel(self, space_name: str, filename: str,
                              monthly_commits: List[Tuple[str, datetime]],
                              space_folder: str) -> Tuple[List[Dict], List[Dict]]:
        """并行处理某个文件的所有月度版本：下载 + 元数据"""
        downloaded_files: List[Dict] = []
        metadata_list: List[Dict] = []

        max_concurrent = min(self.file_workers, len(monthly_commits))
        if max_concurrent <= 0:
            return downloaded_files, metadata_list

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 下载任务
            download_futs = {
                executor.submit(
                    self.download_file_version, space_name, filename, cid, cdate, space_folder
                ): (cid, cdate)
                for cid, cdate in monthly_commits
            }
            # 元数据任务
            meta_futs = {
                executor.submit(self.get_space_metadata, space_name, cid): (cid, cdate)
                for cid, cdate in monthly_commits
            }

            # 收集下载
            for fut in as_completed(download_futs):
                try:
                    res = fut.result()
                    if res:
                        downloaded_files.append(res)
                except Exception:
                    pass

            # 收集元数据
            for fut in as_completed(meta_futs):
                cid, cdate = meta_futs[fut]
                try:
                    meta = fut.result()
                    if meta:
                        meta["source_file"] = filename
                        meta["file_commit_date"] = cdate.isoformat()
                        metadata_list.append(meta)
                except Exception:
                    pass

        return downloaded_files, metadata_list

    def process_single_space(self, space_name: str) -> Dict:
        """处理单个 space：发现文件 -> 取文件历史 -> 月度抽样 -> 下载与元数据"""
        try:
            space_folder = self.create_space_folder(space_name)
            space_data: Dict = {
                "space_name": space_name,
                "space_folder": space_folder,
                "processing_time": datetime.now(timezone.utc).isoformat(),
                "files_data": {},
                "metadata_history": [],
                "status": "processing",
            }

            # 发现候选文件
            target_files = self.discover_target_files(space_name)
            if not target_files:
                target_files = ["app.py", "README.md"]

            for filename in target_files:
                commits = self.get_file_commits(space_name, filename)  # 文件级变更历史（旧->新）
                if not commits:
                    # 该文件在历史上从未存在或未变更
                    continue

                # monthly_commits = self.get_monthly_commits(commits)
                monthly_commits = commits  # 关闭月采样：使用完整提交历史
                if not monthly_commits:
                    continue

                downloaded_files, metadata_list = self.process_file_parallel(
                    space_name, filename, monthly_commits, space_folder
                )

                space_data["files_data"][filename] = {
                    "total_change_commits": len(commits),
                    "monthly_commits_count": len(monthly_commits),
                    "downloaded_files": downloaded_files,
                    "commit_timeline": [(cid, cdate.isoformat()) for cid, cdate in monthly_commits],
                }

                space_data["metadata_history"].extend(metadata_list)

            space_data["status"] = "completed"
            self._save_space_data(space_folder, space_data)
            return space_data

        except Exception as e:
            return {
                "space_name": space_name,
                "status": "failed",
                "error": str(e),
                "processing_time": datetime.now(timezone.utc).isoformat(),
            }

    # ---------- 保存与报告 ----------
    def _save_space_data(self, space_folder: str, space_data: Dict):
        try:
            with open(Path(space_folder) / "space_data.json", "w", encoding="utf-8") as f:
                json.dump(space_data, f, indent=2, ensure_ascii=False, default=str)

            summary = {
                "space_name": space_data["space_name"],
                "total_files": len(space_data["files_data"]),
                "total_versions": sum(len(f["downloaded_files"]) for f in space_data["files_data"].values()),
                "metadata_count": len(space_data["metadata_history"]),
                "status": space_data["status"],
                "processing_time": space_data["processing_time"],
            }
            with open(Path(space_folder) / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")

    def _generate_report(self, results: Dict):
        report_file = Path(self.output_dir) / "collection_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# 🚀 高性能 HuggingFace Spaces 收集报告\n\n")
            f.write(f"**开始时间**: {results['start_time']}\n")
            f.write(f"**结束时间**: {results['end_time']}\n")
            f.write(f"**总处理时间**: {results['total_duration']:.1f} 秒\n\n")
            f.write("## 📊 性能统计\n\n")
            f.write(f"- **总 spaces 数**: {results['total_spaces']}\n")
            f.write(f"- **成功收集**: {results['successful_count']}\n")
            f.write(f"- **收集失败**: {results['failed_count']}\n")
            f.write(f"- **成功率**: {results['success_rate']:.1f}%\n")
            f.write(f"- **并发线程数**: {results['max_workers']}\n")
            if results["total_duration"] > 0:
                rate = results["successful_count"] / results["total_duration"]
                f.write(f"- **处理速率**: {rate:.2f} spaces/秒\n")

    # ---------- 总控 ----------
    def collect_all_spaces(self, csv_path: str) -> Dict:
        print("🚀 启动高性能数据收集器")
        print(f"📊 CSV 文件: {csv_path}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🧵 主线程数: {self.max_workers}")
        print(f"📄 文件处理并发: {self.file_workers}")

        spaces_list = self.load_spaces_from_csv(csv_path)
        if not spaces_list:
            print("❌ 未找到有效的 spaces 列表")
            return {}

        print(f"📋 待处理 spaces 数量: {len(spaces_list)}")

        progress = ThreadSafeProgress(len(spaces_list))
        results: Dict = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "total_spaces": len(spaces_list),
            "max_workers": self.max_workers,
            "successful_spaces": [],
            "failed_spaces": [],
            "spaces_data": {},
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_map = {executor.submit(self.process_single_space, s): s for s in spaces_list}
            for fut in as_completed(fut_map):
                space_name = fut_map[fut]
                try:
                    res = fut.result()
                    results["spaces_data"][space_name] = res
                    if res.get("status") == "completed":
                        results["successful_spaces"].append(space_name)
                        progress.update(success=True)
                    else:
                        results["failed_spaces"].append(space_name)
                        progress.update(success=False)
                except Exception:
                    results["failed_spaces"].append(space_name)
                    progress.update(success=False)

        results.update({
            "end_time": datetime.now(timezone.utc).isoformat(),
            "successful_count": len(results["successful_spaces"]),
            "failed_count": len(results["failed_spaces"]),
            "success_rate": len(results["successful_spaces"]) / len(spaces_list) * 100,
            "total_duration": time.time() - progress.start_time,
        })

        with open(Path(self.output_dir) / "final_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        self._generate_report(results)
        return results


# =======================
# 入口
# =======================
def main():
    """主函数：配置可按需修改"""
    CONFIG = {
        # 推荐使用环境变量： export HF_TOKEN=xxx
        "HF_TOKEN": os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN"),
        "CSV_PATH": "spaces.csv",
        "OUTPUT_DIR": "spaces_data",
        "MAX_WORKERS": 14,    # 主并发（注意 API 限流）
        "FILE_WORKERS": 12,    # 单 space 内文件并发
    }

    print("⚡ 启动高性能 HuggingFace Spaces 收集器")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔥 并发配置: 主 {CONFIG['MAX_WORKERS']} / 文件 {CONFIG['FILE_WORKERS']}")

    try:
        collector = HighPerformanceSpaceCollector(
            hf_token=CONFIG["HF_TOKEN"],
            output_dir=CONFIG["OUTPUT_DIR"],
            max_workers=CONFIG["MAX_WORKERS"],
            file_workers=CONFIG["FILE_WORKERS"],
        )
        t0 = time.time()
        results = collector.collect_all_spaces(CONFIG["CSV_PATH"])
        t1 = time.time()

        print("\n" + "=" * 60)
        print("🎉 收集完成")
        print("=" * 60)
        print(f"📊 总 spaces: {results.get('total_spaces', 0)}")
        print(f"✅ 成功: {results.get('successful_count', 0)}")
        print(f"❌ 失败: {results.get('failed_count', 0)}")
        print(f"📈 成功率: {results.get('success_rate', 0):.1f}%")
        print(f"⏱️  总耗时: {results.get('total_duration', t1 - t0):.1f} 秒")
        if results.get("successful_count", 0) > 0:
            rate = results.get("successful_count", 0) / max(results.get("total_duration", t1 - t0), 1e-6)
            print(f"🚀 处理速率: {rate:.2f} spaces/秒")
        print(f"📂 数据位置: {CONFIG['OUTPUT_DIR']}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()