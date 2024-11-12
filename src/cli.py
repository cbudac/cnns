from lightning.pytorch.cli import LightningCLI
from logging import getLogger


logger = getLogger(__name__)
logger.setLevel('DEBUG')


def cli_main():
    LightningCLI(auto_configure_optimizers=False)

if __name__ == "__main__":
    cli_main()