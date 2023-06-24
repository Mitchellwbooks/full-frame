from multiprocessing import Process


class ModelManager( Process ):
    """
    Abstracts over model versioning
    """

    def __init__(
            self,
            controller_to_model_manager,
            model_manager_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer
    ):
        super().__init__()
        self.controller_to_model_manager = controller_to_model_manager
        self.model_manager_to_controller = model_manager_to_controller
        self.inferencer_to_model_manager = inferencer_to_model_manager
        self.model_manager_to_inferencer = model_manager_to_inferencer

    def run( self ):
        pass

    def async_run(self):
        while True:
            await self.load_inferencer_message()
            await self.load_controller_message()

    async def load_inferencer_message(self):
        pass

    async def load_controller_message(self):
        pass
