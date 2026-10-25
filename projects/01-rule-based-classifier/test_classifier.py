from classifier import classify_customer

assert classify_customer(2500,15,10) == "VIP"

assert classify_customer(1000,15,10) == "Regular"
assert classify_customer(2500,10,10) == "Regular"
assert classify_customer(2500, 15, 30) == "VIP"
assert classify_customer(500, 5, 90) == "Regular"
assert classify_customer(3000, 25, 120) == "At Risk"
# Customer A
assert classify_customer(2500, 15, 31) == "Regular"

# Customer B
assert classify_customer(500, 5, 91) == "At Risk"
try: 
    classify_customer(2500,-15,30)
except ValueError:
    pass
else: 
    raise AssertionError("Expected ValueError for negative orders")

try: 
    classify_customer(500,5,-90)
except ValueError:
    pass
else: 
    raise AssertionError("Expected ValueError for negative days")

try: 
    classify_customer(-3000,25,120)
except ValueError:
    pass
else: 
    raise AssertionError("Expected ValueError for negative dollars")
#Boolean Vaildation Test
try: 
    classify_customer(2500, True, 10)
except TypeError:
    pass
else: 
    raise AssertionError("Expected TypeError for negative dollars")

try: 
    classify_customer(False, 15, 10)
except TypeError:
    pass
else: 
    raise AssertionError("Expected TypeError for negative dollars")
try: 
    classify_customer(2500, 15, True)
except TypeError:
    pass
else: 
    raise AssertionError("Expected TypeError for negative dollars")


assert classify_customer(2500.50, 15, 10) == "VIP"

try:
    classify_customer("2500", 15, 10)
except TypeError:
    pass
else: 
    raise AssertionError("Expected Type Error for string total spent")

try:
    classify_customer(2500, 15.5, 10)
except TypeError:
    pass
else: 
    raise AssertionError("Expected Type Error for decimal orders")

try:
    classify_customer(2500, 15, 10.5)
except TypeError:
    pass
else: 
    raise AssertionError("Expected Type Error for decimal days")

assert classify_customer(2500.50, 15, 10) == "VIP"