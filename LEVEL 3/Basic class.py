class person:

    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, my name is {self.name}.")

class_instance=person('Alice')

greeting=class_instance.greet()

print(greeting)