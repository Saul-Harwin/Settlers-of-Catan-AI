from colorama import init
init()

class Colours:
    def __init__(self):
        self.colours = {
            "red": "\033[91m",
            "green": "\033[92m",
            "blue": "\033[94m",
            "yellow": "\033[93m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
        }
        self.reset = "\033[0m"
    
    def colour_text(self, text, colour):
        return f"{self.colours.get(colour.lower(), '')}{text}{self.reset}"

# Create an instance
colours = Colours()