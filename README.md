# 630-SOUTHWEST-SCHEDULING

-  [Full Report Link](https://github.com/Musiik-fn/630-SOUTHWEST-SCHEDULING/blob/main/630%20Final%20Project%20Report_1%20Signed%20JC.pdf)
-  [Excel file link](https://github.com/Musiik-fn/630-SOUTHWEST-SCHEDULING/tree/main/Linear%20Programming)

## Project Proposal 
Southwest's business model is distinct from that of other US airlines because it uses a point-to-point network. Point-to-point transit is a transportation system in which a plane, bus, or train travels directly to a destination, rather than going through a central hub. The concept of this project is to replicate a simplified and small portion of a point-to-point network using Southwest Airline flight data. 

We will choose 2 distinct sets of nodes, one set of origin nodes and one set of destination nodes. We will look at all the data for a given month. Both the supply and the capacity of the nodes will be predicted using a model such as regression. The objective is to develop a Linear Programming model that optimizes the assignment of flights from these two distinct sets of airports within Southwest Airlines' network. The goal is to minimize total operational delays.

## Phase II: Predictive Analysis
Using publicly available airline data provided by the Bureau of Transportation Statistics (BTS), we aim to predict the following values for Southwest Airlines Co. for June 2025: Origin Node Supply Amounts, Destination Node Demand Amounts, Delays (Node Edges). Our origin airports are BWI, MDW, DAL, DEN, LAS, and the destination airports are LAX, OKC, SAN, SEA, LGA, CHS, DCA, HNL, MIA. Once these values are predicted, we will have the constraints for our point-to-point network as well as the values of node edges (interpretable as the delay cost for sending a flight from origin to destination).

## Phase III: Prescriptive Analytics
With the airport supply, demands, and route delays predicted for June 2025, we aim to use these values to formulate and solve a linear integer programming assignment problem. In this problem, the cost is the predicted delay on each route. Given that our total demand exceeds our supply, we cannot exceed the demand for each destination node, but we must assign all flights from each origin node, and finally we must meet a minimum number of flights for each route.
