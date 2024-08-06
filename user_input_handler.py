class UserInputHandler:
    @staticmethod
    def get_file_paths_from_input():
        file_paths = input("Enter file paths separated by commas: ").split(',')
        file_paths = [file_path.strip() for file_path in file_paths]
        return file_paths
