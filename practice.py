import os

print("hello world")
name = "csaba"
print(name)

print ("5 + 2 =", 5+2)
print ("5 - 2 =", 5-2)
print ("5 * 2 =", 5*2)
print ("5 / 2 =", 5/2)
print ("5 % 2 =", 5%2)
print ("5 ** 2 =", 5**2)
print ("5 // 2 =", 5//2)

quote ="\"quote used"
multi_line_quote = '''just
like everyone else'''

# lists

grocery_list = ['juice', 'tomatoes', 'potatoes', 'bananas']

print("first item", grocery_list[0])
grocery_list[0] = "other juice"
print("first item", grocery_list[0])

print(grocery_list[1:3])

other_events = ['wash car', 'cash check', 'do other stuff']
todos = [other_events, grocery_list]
print(todos[1][1])

grocery_list.append('onions')
print (todos)
grocery_list.insert(1, "pickle")
grocery_list.sort()
grocery_list.reverse()
grocery_list.remove("pickle")

del grocery_list[4]
print (todos)

todos2 = other_events + grocery_list

print(len(todos2))
print(max(todos2))
print(min(todos2))

# Tuples, lists that cant be changed, unless its being converted into a list

pi_tuple = (3,1,4,1,5,9)
new_tuple = list(pi_tuple)
new_list = tuple(pi_tuple)

print (len(pi_tuple))
print (min(pi_tuple))
print (max(pi_tuple))

#dictionaries (in other words maps)

super_villains = {'Fiddler': 'isaac bowin',
                  'captain cold': 'leonard snart',
                  'weather wizard': 'mark mardon',
                  'mirror master': 'sam scudder',
                  'pied piper': 'thomas petherson'}

print (super_villains['captain cold'])
del super_villains['Fiddler']
super_villains['pied piper'] = 'hartley rathaway'
print (len(super_villains))

print (super_villains.get("pied piper"))
print(super_villains.keys())
print(super_villains.values())


#conditional operators

age = 3
if age > 16:
    print('youre ok to drive')
else:
    print('cant drive')

if age >= 21:
    print('ok to trive tractor')
elif age >= 16:
    print ('ok to drive car')
else:
    print("youre not old enough")

if ((age >= 1) and (age <= 18)):
    print ("you get a bithday")
elif((age == 21) or (age >= 65)):
    print ("you get a birthday")
else:
    print ("birthday party")



# looping

for x in range(0, 10):
    print (x)
    print ('\n')

for y in grocery_list:
        print (y)

for x in [2,4,6,8,10]:
            print (x)

num_list = [[1,2,3],[10,20,30],[100,200,300]]
for x in range(0,3):
    for y in range(0,3):
        print (num_list[x][y])

import random
random_num = random.randrange(0,100)

while(random_num != 15):
    print (random_num)
    random_num = random.randrange(0,100)


#functions

def addNumbers(fNum, lNum):
    sumNum = fNum + lNum
    return sumNum

import sys
print(addNumbers(1,4))

print('whats your name?')
name = sys.stdin.readline()
print ('Hello', name)

long_string = "i'll catch you if you fall - the Floor"
print(long_string[0:4])
print(long_string[-5:])
print(long_string[:4] +" be there")
print("%c is my %s letter and my number %d number is %.5f" %
      ('X', 'favorite', 1, .14))
print(long_string.capitalize())
print (long_string.find("Floor"))

#test_file = open("test.txt", "wb")
#print(test_file.mode)
#print (test_file.name)
#test_file.write(bytes("write me to the file\n", 'UTF-8'))
#test_file.close()
#test_file = open("test.txt", "r+")
#test_in_file = test_file.read()
#print(test_in_file)
#os.remove("test.txt")

class Animal:
    __name = ""
    __height = 0
    __weight =0
    __sound =0

#constructor
    def __init__(self, name, height, weight, sound):
        self.__name = name
        self.__height = height
        self.__weight = weight
        self.__sound =sound
#setters/getters
    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    def set_height(self, height):
        self.__height = height

    def get_height(self):
        return self.__height

    def set_weight(self, weight):
        self.__weight = weight

    def get_weight(self):
        return self.__weight

    def set_sound(self, sound):
        self.__sound = sound

    def get_sound(self):
        return self.__sound

    def get_type(self):
        print ("Animal")

    def toString(self):
        return "{} is {} cm tall and {} kilograms and say {}".format(self.__name,
                                                                     self.__height,
                                                                     self.__weight,
                                                                     self.__sound)
cat = Animal('Whiskers',22,10,'meow')
print(cat.toString())

class Dog(Animal):
    __owner = ""
    def __int__(self, name, height, weight, sound, owner):
        self.__owner = owner
        super(Dog, self).__init__(name,height,weight,sound)

    def set_owner(self, owner):
        self.__owner = owner

    def get_owner(self):
        return self.__owner

    def get_type(self):
        print("dog")

    def toString(self):
        return "{} is {} cm tall and {} kilograms and say {} his owner is {}".format(self.__name,
                                                                                     self.__height,
                                                                                     self.__weight,
                                                                                     self.__sound,
                                                                                     self.__owner)

    def multiple_sound (self, how_many = None):
        if how_many is None:
            print(self.get_sound())
        else:
            print (self.get_sound()*how_many)

spot = Dog("Spot", 55, 35,"ruff", "csaba")
print(spot.toString())





