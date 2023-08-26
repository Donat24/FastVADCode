class TrainException(Exception):
    def __init__(self, message = "Fehler beim Training", model = None, batch = None, out=None, loss = None) -> None:
        super().__init__(message)
        self.model = model
        self.batch = batch
        self.out   = out
        self.loss  = loss