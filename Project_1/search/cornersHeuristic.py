corners_set = set(corners)
visited_corners = state[2]
visited_corners_set = set(visited_corners)
corners = corners_set.difference(visited_corners_set)
corners = tuple(corners)

distance = 0
closest = ()
shortest = math.inf
corners_list = list(corners)
s_corner = ()
h = 0
for corner in corners_list:
    distance = ((state[0] - corner[0]) ** 2 + (state[1] - corner[1]) ** 2) ** 0.5
    if distance < shortest:
        shortest = distance
        s_corner = corner
corners_list.remove(s_corner)
h += shortest
corner_to_corner = math.inf
distance_2 = 0
while corners_list:
    for r_corner in corners_list:
        distance_2 = ((s_corner[0] - r_corner[0]) ** 2 + (s_corner[1] - r_corner[1]) ** 2) ** 0.5
        if distance_2 < corner_to_corner:
            corner_to_corner = distance_2
            s_corner = r_corner
    corners_list.remove(s_corner)
    h += corner_to_corner
    corner_to_corner = math.inf