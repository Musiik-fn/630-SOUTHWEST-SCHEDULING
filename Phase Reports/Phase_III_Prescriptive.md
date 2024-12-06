# Introduction
This document outlines the optimization model developed for the analysis of predicted supply and demand across various airports, along with delay predictions for 2025. The methodology, results, and sensitivity analysis are presented to support decision-making and future planning.

# Problem Formulation

## Predicted Supply

| IATA Code | Predicted Supply |
|-----------|------------------|
| BWI       | 375              |
| MDW       | 800              |
| DAL       | 584              |
| DEN       | 834              |
| LAS       | 771              |
| **Total** | **3364**         |

## Predicted Demand

| IATA Code | Predicted Demand |
|-----------|------------------|
| LAX       | 750              |
| OKC       | 248              |
| SAN       | 855              |
| SEA       | 347              |
| LGA       | 402              |
| CHS       | 207              |
| DCA       | 309              |
| HNL       | 60               |
| MIA       | 205              |
| **Total** | **3383**         |

## 2025 Predicted Average Departure Delay (Minutes)

| Origin Airport | Destination Airport | 2025 Predicted Avg Departure Delay (Minutes) |
|----------------|---------------------|----------------------------------------------|
| BWI            | CHS                 | 16.01                                        |
| BWI            | LAX                 | 32.58                                        |
| BWI            | LGA                 | 26.64                                        |
| BWI            | MIA                 | 25.15                                        |
| BWI            | OKC                 | 20.17                                        |
| BWI            | SAN                 | 12.9                                         |
| BWI            | SEA                 | 19.75                                        |
| MDW            | CHS                 | 17.71                                        |
| MDW            | DCA                 | 16.69                                        |
| MDW            | LAX                 | 23.56                                        |
| MDW            | LGA                 | 15.33                                        |
| MDW            | MIA                 | 19.41                                        |
| MDW            | OKC                 | 27.19                                        |
| MDW            | SAN                 | 21.68                                        |
| MDW            | SEA                 | 22.74                                        |
| DAL            | CHS                 | 15.92                                        |
| DAL            | DCA                 | 17.59                                        |
| DAL            | LAX                 | 22.98                                        |
| DAL            | LGA                 | 20.68                                        |
| DAL            | MIA                 | 27.04                                        |
| DAL            | OKC                 | 24.75                                        |
| DAL            | SAN                 | 18.84                                        |
| DAL            | SEA                 | 19.76                                        |
| DEN            | CHS                 | 14.9                                         |
| DEN            | LAX                 | 23.05                                        |
| DEN            | LGA                 | 32.35                                        |
| DEN            | MIA                 | 25.13                                        |
| DEN            | OKC                 | 24.25                                        |
| DEN            | SAN                 | 21.75                                        |
| DEN            | SEA                 | 22.48                                        |
| LAS            | HNL                 | 15.47                                        |
| LAS            | LAX                 | 22.2                                         |
| LAS            | OKC                 | 17.39                                        |
| LAS            | SAN                 | 15.96                                        |
| LAS            | SEA                 | 20.54                                        |

## **Sets**
1. $O$: Set of origin airports (e.g., $\text{BWI, MDW, DAL, DEN, LAS}$)
2. $D$: Set of destination airports (e.g., $\text{LAX, OKC, SAN, SEA, LGA, CHS, DCA, HNL, MIA}$)

---

## **Parameters**
- $\text{Supply}_o$: Predicted supply at origin airport $o$
- $\text{Demand}_d$: Predicted demand at destination airport $d$
- $\text{Delay}_{o,d}$: Predicted average departure delay (minutes) from origin $o$ to destination $d$
- $\text{MinMix}_{o,d} = 0.1 \times \text{Supply}_o$: Minimum flight assignment mix for route $(o,d)$

---

## **Decision Variables**
- $X_{o,d}$: Number of flights (or assigned units) from origin $o$ to destination $d$

---

## **Objective Function**
Minimize the total delay across all routes:

$$
\text{Minimize} \quad Z = \sum_{o \in O} \sum_{d \in D} \text{Delay}_{o,d} \cdot X_{o,d}
$$

---

## **Constraints**

### **1. Supply Constraints**
For each origin airport $o$, the total assigned amount cannot exceed the predicted supply:

$$
\sum_{d \in D} X_{o,d} = \text{Supply}_o \quad \forall o \in O
$$

### **2. Demand Constraints**
For each destination airport $d$, the total assigned amount cannot exceed the predicted demand. Because we have higher demand than supply, setting demand as a minimum would be infeasible.

$$
\sum_{o \in O} X_{o,d} \leq \text{Demand}_d \quad \forall d \in D
$$

### **3. Minimum Flight Assignment Mix**
For each route $(o,d)$, the number of flights assigned must meet the minimum mix constraint, defined by the 10% of the supply node. This is to ensure that a minimum number of flights on the route are satisfied. 

$$
X_{o,d} \geq 0.1 \times \text{Supply}_o \quad \forall (o,d)
$$

### **4. Integrality Constraints**
The decision variables must be integers:

$$
X_{o,d} \in \mathbb{Z} \quad \forall (o,d)
$$

---

## **LP Formulation Summary**

$$
\text{Minimize: } \quad Z = \sum_{o \in O} \sum_{d \in D} \text{Delay}_{o,d} \cdot X_{o,d}
$$

Subject to:

1. $$ \sum_{d \in D} X_{o,d} = \text{Supply}_o \quad \forall o \in O $$
2. $$ \sum_{o \in O} X_{o,d} \leq \text{Demand}_d \quad \forall d \in D $$
3. $$ X_{o,d} \geq 0.1 \times \text{Supply}_o \quad \forall (o,d) $$
4. $$ X_{o,d} \in \mathbb{Z} \quad \forall (o,d) $$

---

# Optimal Solution
- Use Excel Solver to determine the optimal solution to your problem.
- Provide an interpretation of your optimal solution.

| Origin Airport | Destination Airport | 2025 Predicted Avg Departure Delay (Minutes) | Decision Variable | Total Delay from Route | Flight Assignment Mix Minimum |
|----------------|---------------------|----------------------------------------------|-------------------|-------------------------|--------------------------------|
| BWI            | CHS                 | 16.01                                        | 38                | 608.38                 | 38                             |
| BWI            | LAX                 | 32.58                                        | 38                | 1238.04                | 38                             |
| BWI            | LGA                 | 26.64                                        | 38                | 1012.32                | 38                             |
| BWI            | MIA                 | 25.15                                        | 38                | 955.7                  | 38                             |
| BWI            | OKC                 | 20.17                                        | 38                | 766.46                 | 38                             |
| BWI            | SAN                 | 12.9                                         | 147               | 1896.3                 | 38                             |
| BWI            | SEA                 | 19.75                                        | 38                | 750.5                  | 38                             |
| MDW            | CHS                 | 17.71                                        | 80                | 1416.8                 | 80                             |
| MDW            | DCA                 | 16.69                                        | 131               | 2186.39                | 80                             |
| MDW            | LAX                 | 23.56                                        | 80                | 1884.8                 | 80                             |
| MDW            | LGA                 | 15.33                                        | 80                | 1226.4                 | 80                             |
| MDW            | MIA                 | 19.41                                        | 80                | 1552.8                 | 80                             |
| MDW            | OKC                 | 27.19                                        | 80                | 2175.2                 | 80                             |
| MDW            | SAN                 | 21.68                                        | 189               | 4097.52                | 80                             |
| MDW            | SEA                 | 22.74                                        | 80                | 1819.2                 | 80                             |
| DAL            | CHS                 | 15.92                                        | 58                | 923.36                 | 58                             |
| DAL            | DCA                 | 17.59                                        | 178               | 3131.02                | 58                             |
| DAL            | LAX                 | 22.98                                        | 58                | 1332.84                | 58                             |
| DAL            | LGA                 | 20.68                                        | 58                | 1199.44                | 58                             |
| DAL            | MIA                 | 27.04                                        | 58                | 1568.32                | 58                             |
| DAL            | OKC                 | 24.75                                        | 58                | 1435.5                 | 58                             |
| DAL            | SAN                 | 18.84                                        | 58                | 1092.72                | 58                             |
| DAL            | SEA                 | 19.76                                        | 58                | 1146.08                | 58                             |
| DEN            | CHS                 | 14.9                                         | 83                | 1236.7                 | 83                             |
| DEN            | LAX                 | 23.05                                        | 111               | 2558.55                | 83                             |
| DEN            | LGA                 | 32.35                                        | 226               | 7311.1                 | 83                             |
| DEN            | MIA                 | 25.13                                        | 83                | 2085.79                | 83                             |
| DEN            | OKC                 | 24.25                                        | 83                | 2012.75                | 83                             |
| DEN            | SAN                 | 21.75                                        | 165               | 3588.75                | 83                             |
| DEN            | SEA                 | 22.48                                        | 83                | 1865.84                | 83                             |
| LAS            | HNL                 | 15.47                                        | 77                | 1191.19                | 77                             |
| LAS            | LAX                 | 22.2                                         | 463               | 10278.6                | 77                             |
| LAS            | OKC                 | 17.39                                        | 77                | 1339.03                | 77                             |
| LAS            | SAN                 | 15.96                                        | 77                | 1228.92                | 77                             |
| LAS            | SEA                 | 20.54                                        | 77                | 1581.58                | 77                             |
|                |                     |                                              | **Grand Total Delay (Objective)** | **71694.89** |                                |


# Sensitivity Analysis
- Perform sensitivity analysis to explore how changes in key parameters (e.g., coefficients in the objective function, constraint bounds) affect the optimal solution.
- Use tables or charts to present results from the sensitivity analysis effectively.

# Interpretation of Results
- Interpret both the results of your optimization model and the sensitivity analysis.
- Discuss insights gained and how they support decision-making in the context of your research problem.

# Trade-Off Analysis
- Discuss trade-offs between conflicting objectives (e.g., cost vs. quality, efficiency vs. sustainability).

# Future Directions
- Based on your findings, propose actionable recommendations and future directions for further improvement or exploration.
- Discuss how the prescriptive analytics findings could guide decision-making in similar problems or areas.
