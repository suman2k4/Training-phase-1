Income={"Salary":2500,"Freelance":500}

Expense={"Rent":1000,"Food":300,"Transport":250,"Utilities":450}

def Calculate_Income(Income_dict):
    return sum(Income_dict.values())

def Calculate_Expense(Expense_dict):
    return sum(Expense_dict.values())

total_income=Calculate_Income(Income)

total_Expense=Calculate_Expense(Expense)

Balance=total_income-total_Expense

print(f"Income: {total_income},Expense:{total_Expense},Balance:{Balance}")