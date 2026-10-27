"""Unfair Life inference API: a hexagonal (ports-and-adapters) service for social-mobility prediction.

Dependencies point inwards: infrastructure depends on application, which depends on domain, and the
domain depends on nothing. The statistical model can therefore be replaced without touching the
business rules, and the business rules can be tested without the model.
"""
