try:
    a=int(input("Enter the first number: "))
    b=int(input("Enter the second number: "))
    c=a/b
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(c)