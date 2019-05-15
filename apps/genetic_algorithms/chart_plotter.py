import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from deap import gp
from string import Template
from apps.backtesting.utils import in_notebook

from apps.genetic_algorithms.gp_utils import recompute_tree_graph


def draw_tree(individual):
    import networkx as nx
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold')
    nx.draw_networkx_labels(g, pos, labels)

    plt.tight_layout()
    plt.show()


### IPython doge DNA visualizations

def create_jupyter_dna_container(container_name):
    html_template = Template("""
    <div id="$container">< /div>
        <style>
        .node {stroke: #fff; stroke-width: 1.5px;}
        .link {stroke: #999; stroke-opacity: .6;}
        < /style>
    """)

    return html_template.substitute({"container": container_name})


def show_doge_dna(container_name, json_file):

    js_template = Template("""
    // We load the latest version of d3.js from the Web.
    require.config({paths: {d3: "https://d3js.org/d3.v3.min"}});

    require(["d3"], function(d3) {

        // Parameter declaration, the height and width of our viz.
        var width = 800,
            height = 600;

        // Colour scale for node colours.
        var color = d3.scale.category10();

        // We create a force-directed dynamic graph layout.
        // D3 has number of layouts - refer to the documentation.
        var force = d3.layout.force()
            .charge(-120)
            .linkDistance(30)
            .size([width, height]);

        // We select the < div> we created earlier and add an 
        // SVG = Scalable Vector Graphics
        var svg = d3.select("#$container").select("svg")
        if (svg.empty()) {
            svg = d3.select("#$container").append("svg")
                        .attr("width", width)
                        .attr("height", height);
        }

        // We load the JSON network file.
        d3.json("$json_file", function(error, graph) {
            // Within this block, the network has been loaded
            // and stored in the 'graph' object.

            // We load the nodes and links into the force-directed
            // graph and initialise the dynamics.
            force.nodes(graph.nodes)
                .links(graph.links)
                .start();

            // We create a < line> SVG element for each link
            // in the graph.
            var link = svg.selectAll(".link")
                .data(graph.links)
                .enter().append("line")
                .attr("class", "link");

            // We create a < circle> SVG element for each node
            // in the graph, and we specify a few attributes.
            var node = svg.selectAll(".node")
                .data(graph.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", function(d) {
                    // We colour the node depending on the degree.
                    if(d.id == 0) return 10; else return 5;
                })
                //.attr("r", 5)  // radius
                .style("fill", function(d) {
                    // We colour the node depending on the degree.
                    return color(d.name); 
                })
                .call(force.drag);

            // The label each node its node number from the networkx graph.
            node.append("title")
                .text(function(d) { return d.name; });

            //node.append("circle")
            //    .attr("r", 5)  // radius
            //    .style("fill", function(d) {
                    // We colour the node depending on the degree.
            //        return color(d.name); 
            //    });

            //node.append("text")
            //    .attr("dx", 12)
            //    .attr("dy", ".35em")
            //    .text(function(d) { return d.name });


            // We bind the positions of the SVG elements
            // to the positions of the dynamic force-directed graph,
            // at each time step.
            force.on("tick", function() {
                link.attr("x1", function(d) { return d.source.x; })
                    .attr("y1", function(d) { return d.source.y; })
                    .attr("x2", function(d) { return d.target.x; })
                    .attr("y2", function(d) { return d.target.y; });

                node.attr("cx", function(d) { return d.x; })
                    .attr("cy", function(d) { return d.y; });
            });
        });
    });
    """)
    return js_template.substitute({"container": container_name, "json_file": json_file})


def show_tree_container(container_name, json_file):

    js_template = Template("""
    // We load the latest version of d3.js from the Web.
    require.config({paths: {d3: "https://d3js.org/d3.v3.min"}});

    require(["d3"], function(d3) {

        var width = 700;
        var height = 650;
        var maxLabel = 150;
        var duration = 500;
        var radius = 5;

        var i = 0;
        var root;
        
        var tree = d3.layout.tree()
            .size([height, width]);
        
        var diagonal = d3.svg.diagonal()
            .projection(function(d) { return [d.y, d.x]; });

        // We select the < div> we created earlier and add an 
        // SVG = Scalable Vector Graphics
        var svg = d3.select("#$container").append("svg")
            .attr("width", width)
            .attr("height", height)
                .append("g")
                .attr("transform", "translate(" + maxLabel + ",0)");
                
        // We load the JSON network file.
        d3.json("$json_file", function(error, json) {
            // Within this block, the network has been loaded
            // and stored in the 'json' object.
            
            root = json;
            root.x0 = height / 2;
            root.y0 = 0;
            
            root.children.forEach(collapse);


            function update(source) 
            {
                // Compute the new tree layout.
                var nodes = tree.nodes(root).reverse();
                var links = tree.links(nodes);
            
                // Normalize for fixed-depth.
                nodes.forEach(function(d) { d.y = d.depth * maxLabel; });
            
                // Update the nodes…
                var node = svg.selectAll("g.node")
                    .data(nodes, function(d){ 
                        return d.id || (d.id = ++i); 
                    });
            
                // Enter any new nodes at the parent's previous position.
                var nodeEnter = node.enter()
                    .append("g")
                    .attr("class", "node")
                    .attr("transform", function(d){ return "translate(" + source.y0 + "," + source.x0 + ")"; })
                    .on("click", click);
            
                nodeEnter.append("circle")
                    .attr("r", 0)
                    .style("fill", function(d){ 
                        return d._children ? "lightsteelblue" : "white"; 
                    });
            
                nodeEnter.append("text")
                    .attr("x", function(d){ 
                        var spacing = computeRadius(d) + 5;
                        return d.children || d._children ? -spacing : spacing; 
                    })
                    .attr("dy", "3")
                    .attr("text-anchor", function(d){ return d.children || d._children ? "end" : "start"; })
                    .text(function(d){ return d.name; })
                    .style("fill-opacity", 0);
            
                // Transition nodes to their new position.
                var nodeUpdate = node.transition()
                    .duration(duration)
                    .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });
            
                nodeUpdate.select("circle")
                    .attr("r", function(d){ return computeRadius(d); })
                    .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
            
                nodeUpdate.select("text").style("fill-opacity", 1);
            
                // Transition exiting nodes to the parent's new position.
                var nodeExit = node.exit().transition()
                    .duration(duration)
                    .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
                    .remove();
            
                nodeExit.select("circle").attr("r", 0);
                nodeExit.select("text").style("fill-opacity", 0);
            
                // Update the links…
                var link = svg.selectAll("path.link")
                    .data(links, function(d){ return d.target.id; });
            
                // Enter any new links at the parent's previous position.
                link.enter().insert("path", "g")
                    .attr("class", "link")
                    .attr("d", function(d){
                        var o = {x: source.x0, y: source.y0};
                        return diagonal({source: o, target: o});
                    });
            
                // Transition links to their new position.
                link.transition()
                    .duration(duration)
                    .attr("d", diagonal);
            
                // Transition exiting nodes to the parent's new position.
                link.exit().transition()
                    .duration(duration)
                    .attr("d", function(d){
                        var o = {x: source.x, y: source.y};
                        return diagonal({source: o, target: o});
                    })
                    .remove();
            
                // Stash the old positions for transition.
                nodes.forEach(function(d){
                    d.x0 = d.x;
                    d.y0 = d.y;
                });
            }
            
            function nbEndNodes(n)
            {
                nb = 0;    
                if(n.children){
                    n.children.forEach(function(c){ 
                        nb += nbEndNodes(c); 
                    });
                }
                else if(n._children){
                    n._children.forEach(function(c){ 
                        nb += nbEndNodes(c); 
                    });
                }
                else nb++;
                
                return nb;
            }
            
            function computeRadius(d)
            {
                if(d.children || d._children) return radius + (radius * nbEndNodes(d) / 10);
                else return radius;
            }
            
            
            
            function click(d)
            {
                if (d.children){
                    d._children = d.children;
                    d.children = null;
                } 
                else{
                    d.children = d._children;
                    d._children = null;
                }
                update(d);
            }
            
            function collapse(d){
                if (d.children){
                    d._children = d.children;
                    d._children.forEach(collapse);
                    d.children = null;
                }
            }
            
            update(root);
        });
    });
    """)
    return js_template.substitute({"container": container_name, "json_file": json_file})



def write_graph_to_json(individual, json_file_name):
    from deap import gp
    import json

    nodes, edges, labels = gp.graph(individual)
    edited_nodes = [{"name": str(v), "id": int(k)} for (k, v) in labels.items()]
    edited_edges = [{"source": item[0], "target": item[1]} for item in edges]
    with open(json_file_name, 'w') as f:
        json.dump({'nodes': edited_nodes, 'links': edited_edges},
                  f, indent=4, )


"""
var data = {
  "name": "A1",
  "children": [
    {
      "name": "B1",
      "children": [
        {
          "name": "C1",
          "value": 100
        },
        {
          "name": "C2",
          "value": 300
        },
        {
          "name": "C3",
          "value": 200
        }
      ]
    },
    {
      "name": "B2",
      "value": 200
    }
  ]
}
"""

def to_text(node, children_dict, labels):
    if len(children_dict[node]) == 0:
        output = f'''
        {{
            "name": "{labels[node]}",
            "value": "{labels[node]}"
        }}
        '''
    else:
        output = f'''
        {{
            "name" : "{labels[node]}",
            "children": [
                {",".join([to_text(child, children_dict, labels) for child in children_dict[node]])}
            ]
        }}
        '''
    return output


def format_dot_node(node, children_dict, labels):
    if labels[node] == "if_then_else":
        return f'''
            {node} [label="{labels[children_dict[node][0]]}",
                     shape=diamond];     
        '''
    elif labels[node] in ["lt", "gt", "and_", "or_", "xor"]:

        label = f"{labels[children_dict[node][0]]} {labels[node]} {labels[children_dict[node][1]]}"
        return f'''
                    {node} [label="{label}",
                             shape=circle];     
                '''

    else:
        return f'''
            {node} [label="{labels[node]}",
                     shape=circle];     
        '''


def condition_to_string(node, children_dict, labels):
    children = children_dict[node]
    if len(children) == 0:
        return labels[node]
    if len(children) == 1:
        ch1 = condition_to_string(children[0], children_dict, labels)
        return f'{labels[node]} ({ch1})'
    if len(children) == 2:
        ch1 = condition_to_string(children[0], children_dict, labels)
        ch2 = condition_to_string(children[1], children_dict, labels)
        return f'({ch1}) {labels[node]} ({ch2})'
    return "???"


def build_node_style(label):
    label = str(label)
    if label.lower() == "sell":
        return "shape=ellipse, fillcolor=firebrick2, fixedsize=true, width=1, color=white"
    elif label.lower() == "buy":
        return "shape=ellipse, fillcolor=forestgreen, fixedsize=true, width=1, color=white"
    elif label.lower() == "ignore":
        return "shape=ellipse, fillcolor=gold, fixedsize=true, width=1, color=white"
    return ""


def to_dot(node, children_dict, labels):
    children = children_dict[node]
    if len(children) == 1:
        return to_dot(children[0], children_dict, labels)

    if labels[node] == "if_then_else":
        label = condition_to_string(children[0], children_dict, labels)
        children = children[1:]
    else:
        label = labels[node]

    output = f'''{node} [label="{prettify_label(label)}",
                         {build_node_style(label)}];'''
    for i, child in enumerate(children):
        output += to_dot(child, children_dict, labels)
        if i == 0:
            edge_label_setting = 'label=" yes "'
        else:
            edge_label_setting = 'label="    no "'

        output += f"{node} -> {child} [{edge_label_setting}, minlen=3];"
    return output


def prettify_label(label):
    label = str(label).split()
    out = ""
    for part in label:
        part = part.replace("_lt_", " < ")
        part = part.replace("_gt_", " > ")
        if part.startswith("(ARG0)"):
            out += part[len("(ARG0)")+1:]
        elif part == "lt":
            out += " < "
        elif part == "gt":
            out += ">"
        elif part in ["and_", "or_"]:
            out += " " + part[:-1] + " "
        elif part == "xor":
            out += " " + part + " "
        else:
            part = part.strip("()")
            # if part.startswith("(") and part.endswith(")"):
            #    part = part[1:len(part)-1]
            out += part.upper()
    return out


def dot_graph(individual):
    dot_template = """digraph {{
      node [ fontcolor=white,
             shape=rectangle,
             color=gray,
             style=filled,
             fontname="helvetica",
             fillcolor=gray20];
                
      edge [ fontname="helvetica", color=gray40, fontcolor=gray20];
      {}
      }}
    """
    nodes, edges, labels = gp.graph(individual)
    d = recompute_tree_graph(nodes, edges)
    dot_str = dot_template.format(to_dot(0, d, labels))

    return dot_str

def save_dot_graph(individual, out_filename, format='dot'):
    from graphviz import Source
    g = Source(dot_graph(individual))
    g.render(out_filename, format=format)


def get_dot_graph(individual):
    from graphviz import Source
    return Source(dot_graph(individual))


def rewrite_graph_as_tree(individual, json_file_name):
    nodes, edges, labels = gp.graph(individual)
    d = recompute_tree_graph(nodes, edges)
    print(to_text(0, d, labels))
    print(d)


def networkx_graph(individual):
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    import pylab
    pylab.figure(1, figsize=(18, 12))

    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()


class DogeDNACanvas:
    def __init__(self, individual, container_name):
        write_graph_to_json(individual, "tmp.json")
        self.container_name = container_name

    def create_container(self):
        return create_jupyter_dna_container(self.container_name)

    def show(self):
        return show_doge_dna(self.container_name, "tmp.json")

    def show_tree(self):
        return show_tree_container(self.container_name, 'tree.json')
