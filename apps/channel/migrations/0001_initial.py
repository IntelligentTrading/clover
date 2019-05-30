# Generated by Django 2.1.2 on 2019-05-15 13:59

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import unixtimestampfield.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ExchangeData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source', models.CharField(max_length=128)),
                ('data', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
                ('timestamp', unixtimestampfield.fields.UnixTimeStampField()),
            ],
        ),
    ]
