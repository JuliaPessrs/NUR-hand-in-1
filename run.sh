#! /bin/sh

echo "Run handin 1 Julia Pessers"

echo "Download data"
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt 
fi

echo "Run Poisson distribution script ..."
python3 PoissonDistribution.py

echo "Run Vandermond matrix subquestion a"
python3 VandermondeMatrixA.py

echo "Run Vandermond matrix subquestion b"
python3 VandermondeMatrixB.py

echo "Run Vandermond matrix subquestion c"
python3 VandermondeMatrixC.py

echo "Run Vandermond matrix subquestion d"
python3 VandermondeMatrixD.py

echo "Generating the pdf"

pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex