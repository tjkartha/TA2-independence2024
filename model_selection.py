def display_models(models):
    print("Please select a model from the list below:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

def get_user_choice(num_models):
    while True:
        try:
            choice = int(input("Enter the number of the model you want to select: "))
            if 1 <= choice <= num_models:
                return choice
            else:
                print(f"Please enter a number between 1 and {num_models}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_model_code(models, choice):
    model_codes = {
        "Model A": "print('Decision Tree classifier')",
        "Model B": "print('Linear regression')",
        "Model C": "print('Logistic regression')",
        "Model D": "print('Random forest classifier')",
        "Model E": "print('Random forest regression')",
        "Model F": "print('Decision Tree regression')"
    }
    selected_model = models[choice - 1]
    selected_code = model_codes[selected_model]
    return selected_model, selected_code