# DLOps Assignment -2

## 📘 Overview

This repository contains the implementation of **four data contracts** written in **YAML**, based on the **Open Data Contract Standard (ODCS 0.9.3)**.

The assignment focuses on **data quality, governance, schema contracts, and enforcement strategies (circuit breakers)** across multiple real-world domains.

Each contract acts as a **stable interface** between data producers and consumers, preventing downstream failures caused by schema changes or bad data.

---

## 📂 Repository Structure

```text
.
├── rides_contract.yaml        # Scenario 1: Ride-Sharing Platform
├── orders_contract.yaml       # Scenario 2: E-commerce Flash Sale
├── thermostat_contract.yaml   # Scenario 3: IoT Smart Thermostats
├── fintech_contract.yaml      # Scenario 4: FinTech Transactions
└── README.md
