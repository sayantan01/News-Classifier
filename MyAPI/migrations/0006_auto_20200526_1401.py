# Generated by Django 2.2.4 on 2020-05-26 14:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyAPI', '0005_auto_20200526_1354'),
    ]

    operations = [
        migrations.AlterField(
            model_name='news',
            name='content',
            field=models.CharField(max_length=7000),
        ),
    ]
