
class Parent:
    def __init__(self, name: str):
        self.name = name

class Child(Parent):

    def get_name(self):
        self.name = "SPARTA"
        print(self.name)



if __name__=="__main__":
    c = Child(name="JOhn")
    c.get_name()