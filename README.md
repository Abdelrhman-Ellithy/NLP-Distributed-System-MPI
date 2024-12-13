# 📚 NLP-Distributed-System-MPI

An **NLP Model** utilizing the power of **MPI Communicator** to implement a distributed system environment for scalable and efficient data preprocessing and machine learning model training.

---

## 🛠️ Features

- **Parallel Data Preprocessing** 🚀: Uses MPI to distribute and preprocess large datasets across multiple processes, enabling faster execution.
- **Flexible NLP Pipelines** 📖: Includes functions for text cleaning (stopword removal, punctuation stripping, URL elimination, and more).
- **Decision Tree Classification** 🌳: Implements a basic machine learning pipeline with sklearn’s DecisionTreeClassifier.
- **Efficient Dataset Handling** 📊: Supports large-scale datasets through intelligent splitting and gathering using MPI.
- **Cross-platform** 🌍: Compatible with any system supporting MPI and Python.

---

## 📂 Project Structure

```
NLP-Distributed-System-MPI
|-- Project.py    # Main Python script
|-- twitter_training.csv             # Example dataset
```
---

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abdelrhman-Ellithy/NLP-Distributed-System-MPI.git
   cd NLP-Distributed-System-MPI
   ```

2. **Install dependencies**:
   - It’s recommended to use a virtual environment:
     ```bash
     python3 -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Ensure MPI is installed**:
   - For Linux:
     ```bash
     sudo apt-get install mpich
     ```
   - For macOS (using Homebrew):
     ```bash
     brew install open-mpi
     ```
   - Verify installation:
     ```bash
     mpiexec --version
     ```

---

## 🚀 How to Run


1. **Run the MPI script**:
   ```bash
   mpiexec -n <num_processes> python Project.py
   ```
   Replace `<num_processes>` with the number of parallel processes you want to use.
---

## 🧹 Preprocessing Functions

- **remove_stopwords**: Eliminates common stopwords to improve data quality.
- **remove_punc**: Strips punctuation marks.
- **remove_digits**: Removes numerical characters.
- **remove_html_tags**: Cleans HTML content.
- **remove_url**: Filters out URLs.

---

## 🧪 Example Workflow

1. **Load Dataset**: Reads and prepares the dataset for processing.
2. **Preprocess with MPI**: Distributes preprocessing tasks across multiple processes.
3. **Vectorize Text**: Converts preprocessed text into numerical features using sklearn’s CountVectorizer.
4. **Train Model**: Fits a Decision Tree classifier.
5. **Evaluate**: Outputs a confusion matrix and classification report.

---

## 📊 Performance

- **Speedup**: Parallel processing significantly reduces preprocessing time for large datasets.
- **Scalability**: Easily scale the system by increasing the number of processes.

---

## 🛡️ License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🌟 Acknowledgments

- **MPI4Py**: For enabling Python-based MPI implementations.
- **scikit-learn**: For powerful machine learning tools.
- **Pandas**: For efficient data manipulation.
- **nltk**: For NLP-specific preprocessing utilities.

---

## 💡 Future Enhancements

- Integration with advanced classifiers (e.g., Random Forest, Gradient Boosting).
- Adding support for GPU-based preprocessing.
- Extending compatibility with cloud-based environments.

---

Happy coding! 🎉