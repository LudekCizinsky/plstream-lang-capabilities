from termcolor import colored
from time import sleep

def starting(task):
  st = colored("Starting:", "red", attrs=["bold"])
  task = colored(task, "white", attrs=[])
  print(st, task)

def working_on(task):
  st = colored("Working:", "yellow", attrs=["bold", "blink"])
  task = colored(task, "white", attrs=[])
  print(st, task)

def finished(task, time=None):
  st = colored("Finished:", "green", attrs=["bold"])
  task = colored(task, "white", attrs=[])
  if time:
    time = colored(f"({round(time, 2)}s)", "white", attrs=['dark'])
    print(st, task, time)
  else:
    print(st, task)

def error(task):
  st = colored("Error:", "grey", "on_red", attrs=[])
  task = colored(task, "white", attrs=[])
  print(st, task)
