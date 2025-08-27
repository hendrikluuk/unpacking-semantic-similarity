class SmartList(list):
    """
    A custom list class that has constraints and provides useful feedback if you ignore them.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return f"SmartList({super().__repr__()})"

    def append(self, object: list):
        if not isinstance(object, list):
            raise TypeError("You must append a relational domain as a list of objects not as an object.")
        return super().append(object) 
