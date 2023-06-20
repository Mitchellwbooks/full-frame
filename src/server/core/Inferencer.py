from multiprocessing import Process


class Inferencer( Process ):
    """
    Abstracts over model execution
    """

    def __init__(
            self,
            controller_to_inferencer,
            inferencer_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer
    ):
        super().__init__()
        self.controller_to_inferencer = controller_to_inferencer
        self.inferencer_to_controller = inferencer_to_controller
        self.inferencer_to_model_manager = inferencer_to_model_manager
        self.model_manager_to_inferencer = model_manager_to_inferencer

    def run( self ):
        pass
