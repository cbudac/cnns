import importlib
from enum import StrEnum


class Stage(StrEnum):
    FIT = "fit" # is (train + val) kind of super stage
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class Loop(StrEnum):
    # these values are used only in logging from the training and validation steps
    TRAIN = "train"
    VAL = "val"


def create_instance(class_path: str, *args, **kwargs):
    """
    Dynamically creates an instance of a class given its module and class name.

    :param class_path: The path of the class to instantiate.
    :param device: The device to move the instance to.
    :param args: Positional arguments to pass to the class constructor.
    :param kwargs: Keyword arguments to pass to the class constructor.
    :return: An instance of the specified class.
    """

    module_name, class_name = class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)  # Import the module
        cls = getattr(module, class_name)  # Get the class from the module
        return cls(*args, **kwargs) # Instantiate the class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not create instance of {class_name} from {module_name}: {e}")
