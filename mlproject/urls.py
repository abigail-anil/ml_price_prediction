from django.contrib import admin
from django.urls import path, include
 
# Error analysis static files support
from django.conf import settings
from django.conf.urls.static import static
import os
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
]
 
# Error analysis static files support
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)