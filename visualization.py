import matplotlib.pyplot as plt
import seaborn as sns


def practice():

    students = ["Asha", "Rahul", "Priya", "John", "Kiran"]
    marks = [90, 80, 70, 60, 50]

    # Bar Chart
    plt.figure(figsize=(8,5))

    plt.bar(students, marks)

    plt.xlabel("Students")
    plt.ylabel("Marks")
    plt.title("Student Marks Visualization")

    plt.show()


    # Histogram
    sns.histplot(marks, bins=5, kde=True)

    plt.xlabel("Marks")
    plt.title("Marks Distribution")

    plt.show()


practice()

print("Done! Review with CIA for feedback.")