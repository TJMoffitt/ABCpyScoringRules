VERSION=$(shell cat VERSION)
UNITTESTS=$(shell find tests -type f -name '*_tests.py')
MAKEDIRS=$(shell find examples -name Makefile -exec dirname {} \;)
whl_file = abcpy-${VERSION}-py3-none-any.whl

.DEFAULT: help
.PHONY: help clean doc doctest exampletest exampletest_mpi package test uninstall unittest unittest_mpi install reinstall $(MAKEDIRS)

help:
	@echo Targets are: clean, doc, doctest, exampletest, exampletest_mpi, package, uninstall, unittest, unittest_mpi, test

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "*.pkl" -type f -delete
	find . -name "__pycache__" -delete
	find . -name ".#*" -delete
	find . -name "#*#" -delete

$(MAKEDIRS):
	make -C $@

# testing

test: unittest unittest_mpi exampletest exampletest_mpi doctest

unittest:
	@echo "Running standard unit tests.."
	python3 -m unittest discover -s tests -v -p "*_tests.py" || (echo "Error in standard unit tests."; exit 1)
	@# remove temporary files created during testing
	@if test -f net.pth; then rm net.pth; fi
	@if test -f net_with_discard_wrapper.pth; then rm net_with_discard_wrapper.pth; fi
	@if test -f scaler.pkl; then rm scaler.pkl; fi
	@if test -f tmp.jnl; then rm tmp.jnl; fi
	@if test -f journal_tests_testfile.pkl; then rm journal_tests_testfile.pkl; fi

unittest_mpi:
	@echo "Running MPI backend unit tests.."
	mpirun -np 2 python3 -m unittest discover -s tests -v -p "backend_tests_mpi.py" || (echo "Error in MPI unit tests."; exit 1)
	mpirun -np 3 python3 -m unittest discover -s tests -v -p "backend_tests_mpi_model_mpi.py" || (echo "Error in MPI unit tests."; exit 1)

exampletest: $(MAKEDIRS)
	@echo "Testing standard examples.."
	python3 -m unittest -v tests/test_examples.py || (echo "Error in example tests."; exit 1)
	@if test -f scaler.pkl; then rm scaler.pkl; fi
	@if test -f seminn_net.pth; then rm seminn_net.pth; fi
	@if test -f triplet_net.pth; then rm triplet_net.pth; fi
	@if test -f tmp.jnl; then rm tmp.jnl; fi

exampletest_mpi:
	@echo "Testing MPI backend examples.."
	mpirun -np 2 python3 -m unittest -v tests/test_examples_mpi.py || (echo "Error in MPI example tests."; exit 1)

doctest:
	make -C doc html || (echo "Error in documentation generator."; exit 1)

coveragetest: $(MAKEDIRS)  # compile models here as well as we check them for codecov as well.
	@command -v coverage >/dev/null 2>&1 || { echo >&2 "Python package 'coverage' has to been installed. Please, run 'pip3 install coverage'."; exit;}
	# unittests
	@- $(foreach TEST, $(UNITTESTS), \
		echo === Testing code coverage: $(TEST); \
		coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest $(TEST); \
	)
#	 unittest_mpi
	@echo === Testing code coverage: tests/backend_tests_mpi.py
	mpirun -np 2 coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest tests/backend_tests_mpi.py
	@echo === Testing code coverage: tests/backend_tests_mpi_model_mpi.py
	mpirun -np 3 coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest tests/backend_tests_mpi_model_mpi.py
	# exampletest
	@echo === Testing code coverage: tests/test_examples.py
	@coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest tests/test_examples.py
	# exampletest_mpi
	@echo === Testing code coverage: tests/examples_tests_mpi.py
	mpirun -np 2 coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest -v tests/test_examples_mpi.py
	coverage html -d build/testcoverage
	coverage report
	@echo
	@echo Detailed test coverage report under build/testcoverage

# documentation
doc:
	make -C doc html

# packaging
package: whl_file


uninstall:
	pip3 uninstall abcpy


install: whl_file
	pip3 install --user build/dist/$(whl_file)

reinstall: uninstall install


whl_file: clean
	mkdir -p build/dist
	python3 setup.py -v bdist_wheel -d build/dist
	@echo
	@echo "Find" `ls build/dist` "in build/dist/."
