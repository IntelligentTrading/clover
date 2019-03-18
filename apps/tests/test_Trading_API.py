import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
import django
django.setup()
from django.test import TestCase
from apps.portfolio.management.commands.rebalancer import balance_portfolios
import time


class APITestCase(TestCase):

    def test_errors(self):
        NUM_TRIALS = 5
        SLEEP_BETWEEN_TRIALS = 60*6
        errors = 0

        for i in range(NUM_TRIALS):
            print(f'>>>> Running test {i+1}/{NUM_TRIALS} (errors = {errors})')
            output = balance_portfolios()
            if output is not None:
                errors += 1
            if i != NUM_TRIALS-1:
                time.sleep(SLEEP_BETWEEN_TRIALS)

        self.assertEqual(
            errors,
            0)



