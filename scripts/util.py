class ArchList(list):
    """
    A list with a .get method that works like dict.get.
    It's also very ancient and has dark magical powers.
    To defeat it you must locate and destroy its phylactery.
    :3
    """

    def get(self, index: int, default=None) -> Any:
        try:
            return super(ArchList, self).__getitem__(index)
        except IndexError:
            return default