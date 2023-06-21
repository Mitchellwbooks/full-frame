from multiprocessing import Queue

from core.Controller import Controller
from core.Inferencer import Inferencer
from core.ModelManager import ModelManager


def main():

    controller_to_inferencer = Queue()
    inferencer_to_controller = Queue()

    controller_to_model_manager = Queue()
    model_manager_to_controller = Queue()

    inferencer_to_model_manager = Queue()
    model_manager_to_inferencer = Queue()

    controller = Controller(
        controller_to_model_manager,
        model_manager_to_controller,
        controller_to_inferencer,
        inferencer_to_controller
    )
    inferencer = Inferencer(
        controller_to_inferencer,
        inferencer_to_controller,
        inferencer_to_model_manager,
        model_manager_to_inferencer
    )
    model_manager = ModelManager(
        controller_to_model_manager,
        model_manager_to_controller,
        inferencer_to_model_manager,
        model_manager_to_inferencer
    )

    controller.run()
    inferencer.run()
    model_manager.run()


if __name__ == '__main__':
    main()
