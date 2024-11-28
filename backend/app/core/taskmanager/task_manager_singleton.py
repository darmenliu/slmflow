from typing import Optional
import logging
from functools import lru_cache

from app.core.taskmanager.finetune_task_manager import FinetuneTaskManager
from app.core.finetune.finetune_impl_k8s_job import K8sFinetuneService
from app.core.kubeclient.finetune_jobs import FinetuneJobClient
from sqlmodel import Session

logger = logging.getLogger(__name__)

_task_manager_instance: Optional[FinetuneTaskManager] = None

def init_task_manager(
    db_session: Session,
    finetune_image: str = "your-finetune-image:latest",
    namespace: str = "finetune",
    max_concurrent_tasks: int = 5,
    max_tasks_per_user: int = 3
) -> FinetuneTaskManager:
    """
    初始化任务管理器单例
    """
    global _task_manager_instance
    
    if _task_manager_instance is None:
        # 创建K8s任务客户端
        job_client = FinetuneJobClient(
            config_file=None,  # 使用默认配置
            finetune_image=finetune_image,
            default_namespace=namespace
        )
        
        # 创建微调服务
        finetune_service = K8sFinetuneService(
            finetune_job_client=job_client,
            db_session=db_session,
            namespace=namespace
        )
        
        # 创建任务管理器
        _task_manager_instance = FinetuneTaskManager(
            finetune_service=finetune_service,
            max_concurrent_tasks=max_concurrent_tasks,
            max_tasks_per_user=max_tasks_per_user
        )
        
        logger.info("FinetuneTaskManager singleton initialized")
    
    return _task_manager_instance

@lru_cache()
def get_task_manager() -> FinetuneTaskManager:
    """
    获取任务管理器单例
    
    Returns:
        FinetuneTaskManager实例
        
    Raises:
        RuntimeError: 如果任务管理器未初始化
    """
    if _task_manager_instance is None:
        raise RuntimeError("TaskManager not initialized. Call init_task_manager first.")
    return _task_manager_instance 