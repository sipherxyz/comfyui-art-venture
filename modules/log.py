import sys
import logging

if logging.getLogger().hasHandlers():
    logger = logging.getLogger("sd")
else:

    class Log:
        def __init__(self, level=logging.INFO) -> None:
            self.level = level

        def __log(self, level, *args, **kwargs):
            if (level >= self.level):
                print("[ArtVenture]", *args, **kwargs)

        def info(self, *args, **kwargs):
            self.__log(logging.INFO, *args, **kwargs)

        def debug(self, *args, **kwargs):
            self.__log(logging.DEBUG, *args, **kwargs)

        def warning(self, *args, **kwargs):
            self.__log(logging.WARNING, *args, **kwargs)

        def error(self, *args, **kwargs):
            self.__log(logging.ERROR, *args, **kwargs, file=sys.stderr)

    logger = Log()