from django.urls import path, include
from . import views

urlpatterns = [
   # path('admin/', admin.site.urls),
   #path('status/',views.newsTag),
   #path('form/',views.FormAPI),
   path('',views.FormAPI),

]
