class Animal:
    def make_sound(self):
        return "Animal makes a sound"
    
class Dog(Animal):
    def make_sound(self):
        return "woof"
    
instance=Dog()

sound=instance.make_sound()

print(sound)

