from time import time
from random import random, uniform
import csv
from openai import OpenAI
import sys
import os
from collections import namedtuple

# Define classes and named tuples
Node = namedtuple('Node', ['feature', 'left', 'right', 'value'])
Edge = namedtuple('Edge', ['label', 'location1', 'location2', 'distance', 'preference', 'themes'])
Location = namedtuple('Location', ['label', 'latitude', 'longitude', 'themes', 'preference'])
Graph = namedtuple('Graph', ['locations', 'edges', 'edge_labels'])
Trip = namedtuple('Trip', ['startLoc', 'edges', 'location_visits', 'edge_visits'])


print(os.environ.get('OPENAI_API_KEY'))

# Define decision tree classes
class DTree:
    def __init__(self): 
        leaf_node1  = Node(None, None, None, value = 0.99)
        leaf_node2 = Node(None,  None, None, value = 0.65)
        leaf_node3 = Node(None, None, None, value=0.65)
        leaf_node4 = Node(None,  None, None, value = 0.2)
        leaf_node5 = Node(None,  None, None, value = 0.2)
        leaf_node6 = Node(None, None, None, value = 0.2)
        leaf_node7 = Node(None,  None, None, value = 0)
        park_node = Node({"Parks"}, leaf_node6,leaf_node7, None)
        hiking_node = Node({"Hiking"}, leaf_node5, park_node, None)
        environment_node = Node({"Environment"}, leaf_node4,hiking_node, None)
        environment_parks_node = Node({"Environment","Parks"}, leaf_node3, environment_node, None)
        environment_hiking_node = Node({"Environment", "Hiking"}, leaf_node2,environment_parks_node, None)
        self.root_node = Node({"Environment","Hiking","Parks"}, leaf_node1,environment_hiking_node, None)

    def find(self, themes):
        curr = self.root_node
        while not self.isLeaf(curr):
            if curr.feature == themes:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value

    def isLeaf(self, node):
        return node.feature is None
class DTree2:
    def __init__(self): 
        leaf_node1  = Node(None, None, None, value = 1)
        leaf_node2 = Node(None,  None, None, value = 0.79)
        leaf_node3 = Node(None, None, None, value=0.57)
        leaf_node4 = Node(None,  None, None, value = 0.17)
        leaf_node5 = Node(None,  None, None, value = 0.17)
        leaf_node6 = Node(None, None, None, value = 0.17)
        leaf_node7 = Node(None,  None, None, value = .01)
        park_node = Node({"Hiking"}, leaf_node6,leaf_node7, None)
        hiking_node = Node({"Parks"}, leaf_node5, park_node, None)
        environment_node = Node({"Environment"}, leaf_node4,hiking_node, None)
        environment_parks_node = Node({"Environment","Parks"}, leaf_node3, environment_node, None)
        environment_hiking_node = Node({"Environment", "Hiking"}, leaf_node2,environment_parks_node, None)
        self.root_node = Node({"Environment","Hiking","Parks"}, leaf_node1,environment_hiking_node, None)

    def find(self, themes):
        curr = self.root_node
        while not self.isLeaf(curr):
            if curr.feature == themes:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value

    def isLeaf(self, node):
        return node.feature is None


def RoundTripRoadTrip(startLoc, LocFile, EdgeFile, AttractionFile, themes_list, decision_tree, maxTime, x_mph, results_file, required_locatoins, forbidden_locations, useDTree):

    if startLoc in forbidden_locations:
        print(f"The starting location {startLoc} is forbidden.")
        return None

    # Initialize the graph
    g = init_graph(LocFile, EdgeFile, themes_list, forbidden_locations, decision_tree, useDTree)

    # Initialize the starting trip
    curr_rt = Trip(startLoc, [], {}, {})
    q = [curr_rt]
    runtime = 0
    ans = []

    again = True
    t0 = time()

    with open(results_file, 'w') as f: 
        count = 0

        # Main loop
        while q and again: 
            curr_rt = q.pop()

            # If the current trip is empty, add all neighbors to the queue
            if not curr_rt.edges: 
                neighbors = [edge for (loc1, loc2), edge in g.edges.items() if loc1 == curr_rt.startLoc]
                neighbors.sort(key = lambda x: x.preference, reverse = True)
                for edge in neighbors: 
                    # If the edge is within the time limit, add it to the queue
                    if edge.distance < maxTime * x_mph: 
                        new_rt = Trip(curr_rt.startLoc, [], {}, {})
                        add_edge(new_rt, edge)
                        q.append(new_rt)
            else:
                # If the current trip is not empty, add the trip to the answer list if it meets the requirements
                if curr_rt.edges[-1].location2.label == startLoc and check_reqs(curr_rt, required_locatoins, forbidden_locations):
                    if trip_time(curr_rt, x_mph) < maxTime/2: 
                        continue 
                    f = open(results_file, 'a')
                    print_path(curr_rt, x_mph, g, results_file, maxTime, count)
                    f.close()
                    ans.append(curr_rt)

                    # Timekeeping
                    tf = time()
                    runtime += tf - t0
                    t0 = time()

                    again = input("Continue (y/n)? ").lower() == 'y'  
                    count += 1

                    if again: 
                        # If the user wants to continue, sort the queue by preference and keep the top 80%
                        q = sorted(q, key = lambda x: trip_totpref(x, g), reverse = True)[:int(len(q)/1.2)]
                        continue 
                    else: 
                        max_pref = max(trip_totpref(rt, g) for rt in ans)
                        min_pref = min(trip_totpref(rt, g) for rt in ans)
                        tot_pref = sum(trip_totpref(rt, g) for rt in ans)
                        avg_pref = tot_pref / len(ans)
                        avg_time = runtime / len(ans)

                        print_results(max_pref, min_pref, avg_pref, avg_time, len(ans), results_file, curr_rt)

                    
                neighbors = [edge for (loc1, loc2), edge in g.edges.items() if loc1 == curr_rt.edges[-1].location2.label]

                nxt_trips = []
                for edge in neighbors: 
                    new_rt = Trip(curr_rt.startLoc, curr_rt.edges.copy(), curr_rt.location_visits.copy(), curr_rt.edge_visits.copy())
                    add_edge(new_rt, edge)
                    if trip_time(new_rt, x_mph) < maxTime: 
                        nxt_trips.append(new_rt)
                
                nxt_trips.sort(
                    key = lambda x: get_direct_dist(g, curr_rt.startLoc, x.edges[-1].location2.label), reverse= True
                )
                q.extend(nxt_trips)
    return



# Helper function
def init_graph(LocFile, EdgeFile, themes_list, forbidden_locations, decision_tree, useDTree):
    graph = Graph({}, {}, {})

    # Read Locations
    with open(LocFile, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            label, latitude, longitude = row[0], row[1], row[2]
            location = Location(label, float(latitude), float(longitude), set(), 0)
            graph.locations[location.label] = location
    
    #Read Edges
    with open(EdgeFile, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            label, loc_a, loc_b, distance = row[0], row[1], row[2], row[3]
            if loc_a in graph.locations and loc_b in graph.locations and 50.0 < float(distance) < 200.0:
                edge = Edge(label, graph.locations[loc_a], graph.locations[loc_b], float(distance), 0, set())
                edge1 = Edge(label, graph.locations[loc_b], graph.locations[loc_a], float(distance), 0, set())
                add_edge_graph(graph, edge)
                add_edge_graph(graph, edge1)
    
    #Theme List
    for loc in graph.locations.values(): 
        for t in themes_list: 
            if random() < .5: 
                loc.themes.add(t)
    
    # Remove forbidden locations
    if forbidden_locations:
        for location in forbidden_locations:
            if location in graph.locations:
                del graph.locations[location]
        
        for key in list(graph.edges.keys()):
            if key[0] in forbidden_locations or key[1] in forbidden_locations:
                del graph.edges[key]
    
    # Loc Prefs
    t = decision_tree()
    for loc in graph.locations.values():
        if not useDTree: 
            loc = loc._replace(preference = uniform(0,1))
            graph.locations[loc.label] = loc
        else: 
            loc = loc._replace(preference = t.find(loc.themes))
            graph.locations[loc.label] = loc

    # Edge Prefs
    for edge in graph.edges.values(): 
        edge = edge._replace(preference = uniform(0,.1))
        graph.edges[(edge.location1.label, edge.location2.label)] = edge
    
    return graph
    
def add_edge(trip, edge):
    trip.edges.append(edge)
    trip.location_visits[edge.location2.label] = trip.location_visits.get(edge.location2.label, 0) + 1
    trip.edge_visits[(edge.location1.label, edge.location2.label)] = trip.edge_visits.get((edge.location1.label, edge.location2.label), 0) + 1

def add_edge_graph(graph, edge):
    key = (edge.location1.label, edge.location2.label)
    key2 = edge.label
    graph.edges[key] = edge
    graph.edge_labels[key2] = edge

def check_reqs(trip, required_locations, forbidden_locations):
    visited_locations = set()
    for edge in trip.edges:
        visited_locations.add(edge.location1.label)
        visited_locations.add(edge.location2.label)

    if required_locations and not required_locations.issubset(visited_locations):
        return False

    if forbidden_locations and forbidden_locations.intersection(visited_locations):
        return False

    return True

def trip_time(trip, x_mph): 
    total = 0
    for edge in trip.edges:
        total += edge.distance/x_mph
        total += edge.preference*2.5
    return total

def print_path(curr_rt, speed, graph, results_file, maxTime, count): 
    with open(results_file, 'a') as f:
        print(f"Solution {count} | start_location: {curr_rt.startLoc} | max_time: {maxTime} hours | speed: {speed} mph")
        f.write(f"Solution {count} | start_location: {curr_rt.startLoc} | max_time: {maxTime} | speed: {speed}\n")

        for i, edge in enumerate(curr_rt.edges):
            inter_loc_label = edge.location2.label
            edge_time = edge.preference*2.5

            inter_loc_pref = graph.locations[inter_loc_label].preference
            inter_loc_time = inter_loc_pref*2.5

            print(f"{i + 1}. {edge.location1.label} to {edge.location2.label} | label: {edge.label} | edge_pref: {edge.preference:.4f}, edge_time: {edge_time:.2f} hours, inter_loc_pref: {inter_loc_pref:.4f}, inter_loc_time: {inter_loc_time:.2f} hours")
            f.write(
                f"{i + 1}. {edge.location1.label} to {edge.location2.label} | label: {edge.label} | edge_pref: {edge.preference:.4f}, edge_time: {edge_time:.2f} hours, inter_loc_pref: {inter_loc_pref:.4f}, inter_loc_time: {inter_loc_time:.2f} hours\n")

        total_trip_preference = trip_totpref(curr_rt, graph)
        total_trip_distance = sum(edge.distance for edge in curr_rt.edges)
        total_trip_time = trip_time(curr_rt, speed)

        print(f"start_from: {curr_rt.startLoc} | total_trip_preference: {total_trip_preference:.4f} | total_distance: {total_trip_distance:.3f} miles | total_trip_time: {total_trip_time:.2f} hours")
        f.write(f"start_from: {curr_rt.startLoc} | total_trip_preference: {total_trip_preference:.4f} | total_distance: {total_trip_distance:.0f} miles | total_trip_time: {total_trip_time:.2f} hours\n\n")
        count += 1

def trip_totpref(trip, graph): 
    total_pref = 0

    for edge_label, visits in trip.edge_visits.items():
        for i in range(visits):
            if i > 3:
                continue
            total_pref += graph.edges[edge_label].preference * (1 - 0.80) ** (i)

    for location_label, visits in trip.location_visits.items():
        for i in range(visits):
            total_pref += graph.locations[location_label].preference * (1 - 0.80) ** (i)

    return total_pref


def get_direct_dist(graph, loc1, loc2): 
    return ((graph.locations[loc1].latitude - graph.locations[loc2].latitude) ** 2 + (
                    graph.locations[loc1].longitude - graph.locations[loc2].longitude) ** 2) ** 0.5

def isLeaf(node): 
    return node.value is not None

def print_results(max_pref, min_pref, avg_pref, avg_time, count, results_file, curr_rt):
    print("\nSummary:")
    print(f"Total Solutions: {count}")
    print(f"Average Instrumented Runtime: {avg_time} seconds")
    print(f"Maximum TotalTripPreference: {max_pref}")
    print(f"Average TotalTripPreference: {avg_pref}")
    print(f"Minimum TotalTripPreference: {min_pref}")

    narrative = get_story(curr_rt)
    print(narrative)

    # Print summary to screen and write to the file
    with open(results_file, 'a') as f:
        sys.stdout = f  # Redirect standard output to the file

        print("\nSummary:")
        print(f"Total Solutions: {count}")
        print(f"Average Instrumented Runtime: {avg_time} seconds")
        print(f"Maximum TotalTripPreference: {max_pref}")
        print(f"Average TotalTripPreference: {avg_pref}")
        print(f"Minimum TotalTripPreference: {min_pref}")
        print(f"{get_story(curr_rt)}")

    # Reset standard output to the console
    sys.stdout = sys.__stdout__

def user_requirements():
    """
    Prompts the user to provide a set of required locations and a set of forbidden locations 
    that must be part of the road trip. Users enter these as comma-separated lists.
    """
    print("Enter starting location (e.g., 'NashvilleTN'): ")
    starting_location_input = input()
    starting_location = starting_location_input if starting_location_input else 'NashvilleTN'

    print("Enter required locations as a comma-separated list (no spaces, e.g., 'NashvilleTN,MemphisTN'): ")
    required_locations_input = input()
    required_locations = set(required_locations_input.split(',')) if required_locations_input else set()

    print("\nEnter forbidden locations as a comma-separated list (no spaces, e.g., 'LouisvilleKY,ChicagoIL'): ")
    forbidden_locations_input = input()
    forbidden_locations = set(forbidden_locations_input.split(',')) if forbidden_locations_input else set()

    print("\nEnter the maximum time of your trip in hours (e.g., '50'): ")
    max_time_input = input()
    max_time = float(max_time_input) if max_time_input else 50

    return starting_location, required_locations, forbidden_locations, max_time


def get_themes():
    with open('themes.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        themes = []
        for row in reader:
            themes.append(row[0])
    return themes

def get_story(trip): 
    
    story = "You will be given a series of locations on a road trip. Generate a narrative of the roadtrip in an exciting and informative way from the info provided. "
    story += "Here are the legs of the trip: \n"
    for i,edge in enumerate(trip.edges):
        story += f"{i+1}. {edge.location1.label} to {edge.location2.label} \n"
        if edge.themes: 
            story += "This location has the following attributes: "
            story += ", ".join(edge.themes)

    client = OpenAI(
        api_key = "YOUR API KEY GOES HERE"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert travel guide who is very descriptive and helpful. "},
                {"role": "user", "content": story},
            ],

        )
        return response.choices[0].message.content
    except Exception as e: 
        print(e)
        return "I'm sorry, I couldn't generate a story at this time."





if __name__ == '__main__':
    starting_location, required_locations, forbidden_locations, max_time = user_requirements()
    themes_list = ['Environment','Parks','Hiking']
    RoundTripRoadTrip(starting_location, 'Locations.csv', 'Edges.csv','Attractions.csv',themes_list, DTree2 , max_time, 80, "results.txt", required_locations, forbidden_locations, True)
