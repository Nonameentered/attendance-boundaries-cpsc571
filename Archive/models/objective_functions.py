from utils import *
from ortools.sat.python import cp_model
from constants import *


def min_total_segregation(model, objective_terms):
    model.Minimize(sum(objective_terms))


def min_max_segregation(model, objective_terms):
    max_to_min = model.NewIntVar(-(SCALING[0] ** 2), SCALING[0] ** 2, "")
    model.AddMaxEquality(max_to_min, objective_terms)
    model.Minimize(max_to_min)
