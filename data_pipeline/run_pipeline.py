from loguru import logger
from .data_manager import DataManager


def main():
    manager = DataManager()

    try:
        manager.pipeline_sync()
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")


if __name__ == "__main__":
    main()