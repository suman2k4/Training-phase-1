stu_grade=[]

for i in range(5):
    student_name = input("Enter student name: ")
    while True:
        
        try:
           student_grade = float(input("Enter student grade: "))
           if student_grade < 0 or student_grade > 100:
            print("Invalid grade. Please enter a grade between 0 and 100.")
           else:
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    stu_grade.append((student_name,student_grade))

def calculate_average_grade(grade):
    total_grade = sum(student_grade for _,student_grade in grade)
    average_grade = total_grade / len(grade)
    return average_grade

def calculate_passing(grade):
    return sum(student_grade>60 for _,student_grade in grade)

average_grade=calculate_average_grade(stu_grade)
passing=calculate_passing(stu_grade)

print(f"Average:{average_grade:.1f}")
print(f"Passing:{passing}")






   
