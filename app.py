import streamlit as st
import matplotlib.pyplot as plt
import math
import heapq

graph = {
    "Oradea": {"Zerind": 71, "Sibiu": 151},
    "Zerind": {"Oradea": 71, "Arad": 75},
    "Arad": {"Zerind": 75, "Sibiu": 140, "Timisoara": 118},
    "Timisoara": {"Arad": 118, "Lugoj": 111},
    "Lugoj": {"Timisoara": 111, "Mehadia": 70},
    "Mehadia": {"Lugoj": 70, "Dobreta": 75},
    "Dobreta": {"Mehadia": 75, "Craiova": 120},
    "Craiova": {"Dobreta": 120, "Rimnicu Vilcea": 146, "Pitesti": 138},
    "Rimnicu Vilcea": {"Craiova": 146, "Sibiu": 80, "Pitesti": 97},
    "Sibiu": {"Oradea": 151, "Arad": 140, "Fagaras": 99, "Rimnicu Vilcea": 80},
    "Fagaras": {"Sibiu": 99, "Bucharest": 211},
    "Pitesti": {"Rimnicu Vilcea": 97, "Craiova": 138, "Bucharest": 101},
    "Bucharest": {"Fagaras": 211, "Pitesti": 101, "Giurgiu": 90, "Urziceni": 85},
    "Giurgiu": {"Bucharest": 90},
    "Urziceni": {"Bucharest": 85, "Hirsova": 98, "Vaslui": 142},
    "Hirsova": {"Urziceni": 98, "Eforie": 86},
    "Eforie": {"Hirsova": 86},
    "Vaslui": {"Urziceni": 142, "Iasi": 92},
    "Iasi": {"Vaslui": 92, "Neamt": 87},
    "Neamt": {"Iasi": 87},
}

coor = {
    "Oradea": (145, 25),
    "Zerind": (95, 65),
    "Arad": (75, 155),
    "Timisoara": (95, 250),
    "Lugoj": (145, 290),
    "Mehadia": (165, 330),
    "Dobreta": (145, 370),
    "Sibiu": (255, 165),
    "Fagaras": (390, 165),
    "Rimnicu Vilcea": (275, 220),
    "Craiova": (280, 370),
    "Pitesti": (355, 260),
    "Bucharest": (515, 300),
    "Giurgiu": (500, 370),
    "Urziceni": (580, 290),
    "Hirsova": (660, 290),
    "Eforie": (695, 340),
    "Vaslui": (650, 210),
    "Iasi": (630, 110),
    "Neamt": (515, 70),
}

scale = 140 / math.hypot(255 - 75, 165 -155)

def heuristik(a, b):
    (x1, y1) = coor[a]
    (x2, y2) = coor[b]
    return math.hypot(x2 - x1, y2 - y1) * scale

def reconstruct_path(came_from, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path

def path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    return cost

def greedy_bfs(start, goal):
    frontier = [(heuristik(start, goal), start)]
    came_from = {start: None}
    visited = set()
    expended = []

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expended.append(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return {
                "found": True,
                "path": path,
                "cost": path_cost(path),
                "expanded": expended,
            }

        for neighbor in graph.get(current, {}):
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristik(neighbor, goal), neighbor))
    
    return {
        "found": False,
        "path": [],
        "cost": None,
        "expanded": expended,
    }

def a_star(start, goal):
    frontier = [(heuristik(start, goal), start)]
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()
    expended = []

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expended.append(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return {
                "found": True,
                "path": path,
                "cost": path_cost(path),
                "expanded": expended,
            }

        for neighbor, weight in graph.get(current, {}).items():
            tentative_g = g_score[current] + weight
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristik(neighbor, goal)
                heapq.heappush(frontier, (f_score, neighbor))

    return {
        "found": False,
        "path": [],
        "cost": None,
        "expanded": expended,
    }

def map(result=None):
    fig, ax = plt.subplots(figsize=(10, 10 * 460 / 760))
    fig.patch.set_facecolor("#191D20")
    ax.set_facecolor("#D9D4D8")

    path = result.get("path", []) if result else []
    expanded = result.get("expanded", []) if result else []
    routes = {tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)}

    drawn = set()
    for city, neighbors in graph.items():
        x1, y1 = coor[city]
        for neighbor, weight in neighbors.items():
            key = tuple(sorted((city, neighbor)))
            if key in drawn:
                continue
            drawn.add(key)
            x2, y2 = coor[neighbor]
            is_route = key in routes
            ax.plot([x1, x2], [-y1, -y2], color="#FF0000" if is_route else "#FFFFFE", linewidth=3 if is_route else 1.3, zorder=2 if is_route else 1)
            ax.text((x1 + x2) / 2, -(y1 + y2) / 2 + 6, str(weight), fontsize=8, color="#FFFFFF", ha="center")

    for city, (x, y) in coor.items():
        is_route = city in path
        is_explored = city in expanded
        is_endpoint = city == path[0] or city == path[-1] if path else False

        face = "#FF0000" if is_route else ("#2F6F62" if is_explored else "#9B9077")
        edge = "#FF0000" if is_route else ("#2F6F62" if is_explored else "#26241F")
        size = 220 if is_endpoint else 140

        ax.scatter(x, -y, s=size, facecolor=face, edgecolor=edge, linewidth=2.2 if is_route else 1.3, zorder=3)
        ax.text(x + 10, -y, city, fontsize=9.5, color="#FFFFFF", va="center")

    ax.set_xlim(0, 760)
    ax.set_ylim(-460, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig

st.set_page_config(page_title="Romania Problem", layout="wide")
st.markdown(
    """
    <h1 style="text-align: center; color: #0D9488;">Romania Problem</h1>
    <p style="text-align: center; color: #708090;"> Greedy Best-First Search and A* Search on the Romania Map</p>
    <p style="text-align: center; color: #708090;"> Dewa Ngakan Putu Sunyananda Triyanca - 5025251152</p>
    """,
    unsafe_allow_html=True,
)

cities = sorted(graph.keys())
col_map, col_controls = st.columns([1.6, 1])

with col_controls:
    st.subheader("Pencarian Path")
    start = st.selectbox("Kota Awal", cities, index=cities.index("Arad"))
    goal = st.selectbox("Kota Tujuan", cities, index=cities.index("Bucharest"))
    algorithm = st.selectbox("Algoritma Pencarian", ["Greedy Best-First Search", "A* Search"])
    run = st.button("Mulai")

    result = None
    if run:
        if start == goal:
            st.warning("Kota awal dan kota tujuan tidak boleh sama.")
        else:
            result = greedy_bfs(start, goal) if algorithm == "Greedy Best-First Search" else a_star(start, goal)

            if not result["found"]:
                st.error("Tidak ditemukan jalur dari kota awal ke kota tujuan.")
            else:
                st.subheader("Hasil")
                m1, m2 = st.columns(2)
                m1.metric("Total Jarak (Cost)", f"{result['cost']} km")
                m2.metric("Jumlah Kota Diperiksa", f"{len(result['expanded'])} kota")

                st.write("**Jalur yang ditemukan:** " + " → ".join(result["path"]))
                st.write("**Urutan kota diperiksa:** " + " → ".join(result["expanded"]))

with col_map:
    st.subheader("Romania Map")
    fig = map(result if run and result["found"] and result["path"] else None)
    st.pyplot(fig, use_container_width=True)