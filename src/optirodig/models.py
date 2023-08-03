from django.db import models
from phonenumber_field.modelfields import PhoneNumberField
# Create your models here.

class Steel(models.Model):
    """Steel properties"""
    name = models.CharField(verbose_name='Steel Name', max_length=50)
    C = models.DecimalField(verbose_name='C', max_digits= 10, decimal_places=4, default=0)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=4, default=0)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=4, default=0)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=4, default=0)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=4, default=0)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=4, default=0)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=4, default=0)

    def __str__(self) -> str:
        return self.name
    
class Schrott(models.Model):
    """scrap information => company, price, menge"""
    name = models.CharField(verbose_name='Schrott Name', max_length=50)
    company = models.CharField(verbose_name='Company Name', max_length=50,null=False, blank=False)
    quantity = models.IntegerField(verbose_name='Quantity', default=0)  
    price = models.DecimalField(verbose_name='Price', max_digits=10, decimal_places=4, default=0)
    
    def __str__(self) -> str:
        return self.name
    
class GiessereiSchrott(models.Model):
    """The kreislauf schrott for gießerei"""
    name = models.CharField(verbose_name="Schrott Name", max_length=50)
    giesserei = models.CharField(verbose_name="Giesserei Name", max_length=50, null=False, blank=False)
    quantity = models.IntegerField(verbose_name="Quantity", default=0)
    
    def __str__(self) -> str:
        return self.name
    
class SchrottChemi(models.Model):
    """The chemi components analysis for every schrott"""
    name = models.CharField(verbose_name='Schrott Name', max_length=50)
    C = models.DecimalField(verbose_name='C', max_digits= 10, decimal_places=4, default=0)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=4, default=0)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=4, default=0)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=4, default=0)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=4, default=0)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=4, default=0)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=4, default=0)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=4, default=0)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=4, default=0)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=4, default=0)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=4, default=0)
    Nb = models.DecimalField(verbose_name='Nb', max_digits=10, decimal_places=4, default=0)
    Al = models.DecimalField(verbose_name='Al', max_digits=10, decimal_places=4, default=0)
    Cu = models.DecimalField(verbose_name='Cu', max_digits=10, decimal_places=4, default=0)
    N = models.DecimalField(verbose_name='N', max_digits=10, decimal_places=4, default=0)
    
    def __str__(self) -> str:
        return self.name
    
class Handler(models.Model):
    """ Schrott und Legierung händler"""
    name = models.CharField(verbose_name='Firma Name', max_length=100)
    telephone = PhoneNumberField(verbose_name='Telefon')
    email = models.EmailField(verbose_name='Email')
    street = models.CharField(verbose_name='Straße', max_length=100)
    plz = models.CharField(verbose_name='PLZ', max_length=5)
    city = models.CharField(verbose_name='Stadt', max_length=100)

    def __str__(self):
        return self.name

class SchrottWerkstoffe(models.Model):

    handler = models.ForeignKey(Handler, verbose_name='Händler',
                                on_delete=models.CASCADE, related_name='schrotten')

    FORM_CHOICES= ((1, 'Stueckschrott'),
                   (2,'Rohbloecken'),(3,'Paketen'), (4,'Draht'),
                   (5,'Masseln'), (6,'Granalien'), (7,'Gratschrott'), (8,'Hackschrott'))

    form = models.SmallIntegerField(verbose_name='Form', choices=FORM_CHOICES)

    SPEZ_CHOICES = ((1,'Schnellstahlschrott_kobaltfrei'),
                      (2,'Schnellstahlschrott_kobaltlegiert'),
                      (3,'Tiefzieh_stanzabfaelle'),
                      (4,'Cr_17_Or_Cr_13'),
                      (5,'Cr_Ni_Schrott'),
                      (6,'Kaltarbeitsstahl'),
                      (7,'Warmarbeitsstahl'),
                      (8,'Cr_Ni_148xx'))

    spezifikation = models.SmallIntegerField(verbose_name='Spezifikation', choices=SPEZ_CHOICES)

    upload_file = models.FileField(verbose_name='Datei')
    # upload_file = models.CharField(verbose_name='Datei', max_length=200)

    upload_datetime = models.DateTimeField(verbose_name='Datetime hochladen', auto_now_add=True)


    def __str__(self):
        return f"{self.handler}_{self.get_spezifikation_display()}_{self.get_form_display()}_{self.upload_datetime.strftime('%Y-%m-%d %H:%M')}"

class Tiefzieh_stanzabfaelle(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Tiefzieh_stanzabfaelle')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Schnellstahlschrott_kobaltfrei(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Schnellstahlschrott_kobaltfrei')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=5)
    A_s = models.DecimalField(verbose_name='As', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Schnellstahlschrott_kobaltlegiert(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Schnellstahlschrott_kobaltlegiert')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=5)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=5)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=5)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Cr_17_Or_Cr_13(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Cr_17_Or_Cr_13')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=5)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Cr_Ni_Schrott(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Cr_Ni_Schrott')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Cu = models.DecimalField(verbose_name='Cu', max_digits=10, decimal_places=5)
    Ti = models.DecimalField(verbose_name='Ti', max_digits=10, decimal_places=5)
    Nb = models.DecimalField(verbose_name='Nb', max_digits=10, decimal_places=5)
    B = models.DecimalField(verbose_name='B', max_digits=10, decimal_places=5)
    Al = models.DecimalField(verbose_name='Al', max_digits=10, decimal_places=5)
    N2 = models.DecimalField(verbose_name='N2', max_digits=10, decimal_places=5)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=5)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    N = models.DecimalField(verbose_name='N', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Kaltarbeitsstahl(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Kaltarbeitsstahl')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=5)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=5)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=5)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=5)
    Cu = models.DecimalField(verbose_name='Cu', max_digits=10, decimal_places=5)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=5)
    Ti = models.DecimalField(verbose_name='Ti', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Warmarbeitsstahl(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Warmarbeitsstahl')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Cr = models.DecimalField(verbose_name='Cr', max_digits=10, decimal_places=5)
    Ni = models.DecimalField(verbose_name='Ni', max_digits=10, decimal_places=5)
    Cu = models.DecimalField(verbose_name='Cu', max_digits=10, decimal_places=5)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=5)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=5)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Ti = models.DecimalField(verbose_name='Ti', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe

class Cr_Ni_148xx(models.Model):
    spezifikation = models.ForeignKey(SchrottWerkstoffe, verbose_name='Spezifikation', on_delete=models.CASCADE, related_name='Cr_Ni_148xx')
    Werkstoffe = models.CharField(verbose_name='Werkstoffe', max_length=100)
    C = models.DecimalField(verbose_name='C', max_digits=10, decimal_places=5)
    Si = models.DecimalField(verbose_name='Si', max_digits=10, decimal_places=5)
    Mn = models.DecimalField(verbose_name='Mn', max_digits=10, decimal_places=5)
    P = models.DecimalField(verbose_name='P', max_digits=10, decimal_places=5)
    S = models.DecimalField(verbose_name='S', max_digits=10, decimal_places=5)
    Mo = models.DecimalField(verbose_name='Mo', max_digits=10, decimal_places=5)
    V = models.DecimalField(verbose_name='V', max_digits=10, decimal_places=5)
    W = models.DecimalField(verbose_name='W', max_digits=10, decimal_places=5)
    Co = models.DecimalField(verbose_name='Co', max_digits=10, decimal_places=5)
    Cu = models.DecimalField(verbose_name='Cu', max_digits=10, decimal_places=5)
    N2 = models.DecimalField(verbose_name='N2', max_digits=10, decimal_places=5)
    Menge = models.DecimalField(verbose_name='Menge', max_digits=10, decimal_places=5)

    def __str__(self):
        return self.Werkstoffe
