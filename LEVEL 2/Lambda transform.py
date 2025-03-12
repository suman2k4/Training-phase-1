list1=[1,2,3,4,5]

filtered_even=filter(lambda x:x%2==0,list1)

even_numbers=list(filtered_even)

cubed_even=map(lambda x:x**3,even_numbers)

final_result=list(cubed_even)

print("the filtered even numbers:",final_result)