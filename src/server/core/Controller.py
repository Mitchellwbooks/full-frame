from multiprocessing import Process


class Controller( Process ):
    """
    Abstracts User Files from rest of program
    """

    def __init__(
            self,
            controller_to_model_manager,
            model_manager_to_controller,
            controller_to_inferencer,
            inferencer_to_controller
    ):
        super().__init__()
        self.controller_to_model_manager = controller_to_model_manager
        self.model_manager_to_controller = model_manager_to_controller
        self.controller_to_inferencer = controller_to_inferencer
        self.inferencer_to_controller = inferencer_to_controller

    def run( self ):
        pass
