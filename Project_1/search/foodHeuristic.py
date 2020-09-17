position, foodGrid = state
"*** YOUR CODE HERE ***"
current = position
food_coordinates = foodGrid.copy().asList()
h = 0
closest_food = ()
shortest_distance = math.inf
while food_coordinates:
    for food in food_coordinates:
        manhattan = abs(current[0] - food[0]) + abs(current[1] - food[1])
        if manhattan < shortest_distance:
            shortest_distance = manhattan
            closest_food = food
    current = closest_food
    food_coordinates.remove(closest_food)
    h += shortest_distance
    shortest_distance = math.inf
return h