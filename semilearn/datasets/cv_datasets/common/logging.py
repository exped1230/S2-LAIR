import logging

from semilearn.datasets.cv_datasets.common.meta_data import PROJECT_NAME
from semilearn.datasets.cv_datasets.common.utils import Singleton
# from common.meta_data import PROJECT_NAME
# from common.utils import Singleton


@Singleton
def set_logging(level='INFO', name=PROJECT_NAME):
    FORMAT = '[{0} %(asctime)s %(filename)s %(lineno)d %(levelname)s] %(message)s'.format(name)
    logging.basicConfig(level=level,
                        datefmt='%m/%d %H:%M:%S',
                        format=FORMAT)
