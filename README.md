
# Gail-Donkeycar 
Implementing deep IRL and GAIL based techniques in Donkeycar.

# Author 
Harshita Chaudhary  
Minh Dang  

# Requirements
Python version:  
Library Packages:  

# Using Expert Trajactories
1. Generating Expert Trajactories:
2. Training model with Expert Trajactories:
3. Result:

# TODO
| Metrics		    | Values		 |
| -----------   | -----------|
| generatorloss	| 0.96885		 |
| expertloss	  | 0.30522		 |
| entropy		    | 0.38185		 |
| entropyloss	  | -0.00038	 |
| generatoracc	| 0.58789		 |
| expertacc		  | 0.83398		 |


# Using Human Input
1. Collecting data from the simulator:
2. Preprocess images:
3. Training model with Segmented Images:
4. Result:

| Metrics		    | Values		 |
| -----------   | -----------|
| generatorloss	| 0.96885		 |
| expertloss	  | 0.30522		 |
| entropy		    | 0.38185		 |
| entropyloss	  | -0.00038	 |
| generatoracc	| 0.58789		 |
| expertacc		  | 0.83398		 |

# Acknowledgements
The base source code has been taken from stable-baselines repository: https://github.com/hill-a/stable-baselines.  
Segmentation idea has been taken from this article: https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html.  
