from langchain.prompts import PromptTemplate

template = """This is a question-answering system over a corpus of PDF manuals documents defined by the customer.
The documents include operational instructions and manuals .

Given chunks from multiple documents and a question, create an answer to the question that references those documents as "SOURCES".

- If the question asks about the system's capabilities, the system should respond with some version of "This system can answer questions based provided PDF manuals documents for a customer household.". The answer does not need to include sources.
- If the answer cannot be determined from the chunks or from these instructions, the system should not answer the question. The system should instead return "No relevant sources found".
- Chunks are taken from the middle of documents and may be truncated or missing context.
- Documents are not guaranteed to be relevant to the question.

QUESTION: How do I make espresso?
=========
Content: Platzieren  Sie die Tasse unter den Kaffeeauslauf (bei Zubereitung von zwei Tassen jeweils eine Tasse unter \njeden Kaffeeauslauf).  Jetzt stellen Sie den Brühgruppen -Bedienhebel nach oben und die Kaffeezubereitung \nbeginnt.  \n \nBei d er CLASSIKA PID wird Ihnen nun auf dem PID -Display der Timer mit Sekundenangabe angezeigt.  In der \nRegel dauert ein Bezug von Espresso ca. 23 – 25 Sekunden. Die Füllmenge eines Epressos liegt bei 25 – 30 ml. \nIst die gewünschte Füllmenge erreicht,  muss der Brüh gruppen -Bedienhebel wieder nach unten gestellt werden. \nAus der unteren Öffnung des Brühgruppenzylinders entladen sich Restdruck/Restwasser in die \nWasserauffangschale.  \n \n Vorsicht!  \nWird der Brühgruppen -Bedienhebel nach der Kaffeezubereitung nicht ganz nac h unten gestellt, \nspritzen bei Heraus nahme des Siebträgers aus der Kaffeebrühgruppe Heißwasser und \nKaffeesud. Dies kann zu Verletzungen führen.  \n \n Wichtig  \nEin optimales Kaffee -Ergebnis ist nur mit frisch gemahlenen Bohnen möglich.  \nErst mit dem richtigen / fei nen Mahlgrad und dem richtigen A npressen des Kaffee mehls  steigt das \nPumpenmanometer.  \n6.5 Heißwasserentnahme
3.1 Two-stage prompting
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

Content: Es besteht die Möglichkeit, das PID -Display auszuschalten: Dies geschieht, indem Sie die + Taste gedrückt \nhalten, bis sich das Display ausschaltet. Es erscheint ein Punkt auf dem Display, welcher Ihnen zeigt, dass \ndie Maschine eingeschaltet ist. Durch erneutes Drücken der + Taste wird das Display wieder eingeschaltet.  \n6.4 Zubereitung von Kaffee  \nVerwenden Sie bitte den F ilterträger  und das entsprechende kleinere Sieb (Eintassensieb) für die Zubereitung \neiner Tasse und das große Sieb (Zweitassensieb) für die Zubereitung  von zwei Tassen.  \nEs ist wichtig, dass das Sieb fest in den Filterträger einges etzt ist.  Befüllen Sie das Sieb mit Kaffeemehl mit \nder richtigen Mahlung für Espresso . Als Richtlinie zur Füllmenge dient die Marki erung im Sieb des \nFiltertägers.  Jetzt drücken Sie das Kaffeemehl mit einem  Tamper an, dann den Siebträger fest in die \nBrühg ruppe einsetzen.   \n \nPlatzieren  Sie die Tasse unter den Kaffeeauslauf (bei Zubereitung von zwei Tassen jeweils eine Tasse unter \njeden Kaffeeauslauf).  Jetzt stellen Sie den Brühgruppen -Bedienhebel nach oben und die Kaffeezubereitung \nbeginnt.
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

Content: ENGLISH  \nTranslation of the German original user manual    37 Cappuccino preparation step by step  \n1. Prepare a portion of espresso using a cappuccino cup.  \n2. Froth milk in a separate  container . \n3. Fill the cup with the espresso and the frothed milk. Do not just pour the milk , but “shake” it into the cup. \nIf nece ssary, use a spoon to scoop the milk into the cup.  \n12. RECOMMENDED ACCESSORIES  \n• Blind filter for brew group cleaning (included in delivery)  \n• Detergent item number: PAV9001034 for brew group cleaning with the blind filter  \n• Descal ing powder item number : PAV9001040  for prophylactic descaling  \n \nFor a perfect coffee result , a good espresso coffee machine and coffee grinder are as important as a good \ncoffee  bean . Our professional espresso coffee machines and grinders are the perfect combination  to \nachieve this result.  \nThe knock -out perfectly complements your espresso coffee machine and your grinder.  \n \n   \n \nC-Manuale 54 grinder  Knockbox ( round ) Knockbox Slim (drawer)  \n   \nTamper, flat or convex  Tamp ing station  Milk pitcher'
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf
=========
FINAL ANSWER: To make espresso, fill the ground coffee with the right grind for espresso into the filter and compress it with a tamper. Clamp the portafilter firmly into the brew group and place a cup under the spout. Activate the brew lever to start the brewing process, aiming for a brewing time of around 23 to 25 seconds until the desired volume of approximately 25 to 30 ml is reached.
SOURCES: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

QUESTION: How do you make a cappuccino with the CLASSIKA PID?
=========
Content: Liebe Kundin, lieber Kunde , \n \nmit der CLASSIKA  PID haben Sie eine sehr gute Wahl getroffen.  Wir wünschen Ihnen viel Freude an Ihrer \nMaschine und vor allem an der Zubereitung von Espresso und Cappuccino . \nWir bitten Sie, diese Bedienungsanleitung vor Gebrauch der Maschine sorgfältig durchzulesen und zu \nbeachten. Sollte der eine oder andere Punkt Ihnen nicht klar und verständlich sein, oder benötigen Sie \nweitere Informationen , so bitten wir Sie, vor der Inbe triebnahme mit Ihrem Händler Kontakt aufzunehmen.  \nBewahren Sie diese Bedienungsanleitung an einem sicheren Platz griffbereit auf, um bei eventuellen \nProblemen auf diese zurückgreifen zu können.  \n \n \nDear customer,  \n \nWith the CLASSIKA  PID, you have purchased an espresso coffee machine  of the highest quality . We thank \nyou for your choice and wish you a lot of pleasure while preparing perfect espresso and cappuccino with \nyour espresso coffee machine.  \nPlease r ead the instruction manual carefull y before using your new  machine.  \nIf you have any further questions or if you require any further information, please contact your local \nspecialised dealer before starting up the espresso coffee machine.  \nPlease keep the instruction manual within reach  for future reference.  \n \n \n \n \n   \n ECM  Espresso Coffee Machines  \nManufacture GmbH  \nIndustriestraße 57 -61 \n69245 Bammental  \nDeutschland/Germany  \n \nTel.:  +49 (0) 6223 9255 - 0 \nE-Mail: info@ecm.de
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

Content: ENGLISH  \nTranslation of the German original user manual    23 \n2.2 Proper use  \nThe CLASSIKA PID has to be used  for the preparation  of coffee, hot water and steam  only. The machine is \nnot intended for commercial use.  \nUse of the machine other than for the above mentioned purpose will void warranty . The manufacturer cannot \nbe held responsible for damages due to unsuitable use of the mac hine and is not liable for recourse.  \n \n This appliance is intended to be used in household and similar applications such as:  \n• staff kitchen areas in shops, offices and other working environments  \n• farm houses  \n• by clients in hotels, motels and other residential type environments  \n• bed and breakfast type environments  \n3. MACHINE DESCRIPTION  \n3.1 Machine parts  \nCLASSIKA PID   \n  \n1 Cup heater, detachable  \n2 Fresh water tank  \n(under the cup heater)  \n3 Steam/ hot water dispenser  \n4 Brew group  \n5 Pump pressure gauge  \n6 PID-Display  \n7 Portafilter  \n8 Power switch  \n9 Green control lamp to indicate On/Off mode and \nwater shortage in the water tank  \n10 Orange control lamp for heating element  \n11 Switch pump/ hot water  \n12 Orange control lamp for steam  \n13 Steam switch  \n14 Brew group lever  \n15 Drip tray  \n \n \n Caution!  \nDanger of injury: The following parts are hot or may become hot:  \n• area around the steam/ hot water handle   \n• steam/ hot water tube  \n• portafilter  \n• brew group  \n• body (upper part and side frames)  \n3.2 Technical data  \nVoltages:  EU: 230 V  \n  UK: 230 V  \n  New Zealand: 230 V  \n  Australia: 230 V  \n  US: 120 V  \n  Japan: 100 V  \nFrequency : EU: 50 Hz  \n  UK: 50 Hz'
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

Content: ENGLISH  \nTranslation of the German original user manual    25  Before preparing the first coffee, please rinse the machine by extracting about 5 -8 l from  \nthe brew group and the hot water wand. When the machine is switched off, more hot water  \ncan be dispensed. See also chapter 6. 5 Dispensing of hot water.  \n Important!  \nThe PID -control assists the machine in keeping a constant boiler temperature. This means  \nthat the machine is heated continuously and that the orange control lamp blinks every  \nsecond. The boiler temperature is indicated on the PID-display.  \n6. USE OF THE MACHINE  \n6.1 Prepar ation of the machine  \nThe switched off machine is to be  placed  into operation as follows:  \n1. Make sure that there is sufficient water in the water tank.  Refill water if necessary.  \n2. Make sure that the switches pump/hot water and steam are in  the lower position, the power switch is in \nlower position  and the steam/ hot water valve is closed.  \n3. If the boiler temperature is below 40°C when the machine is switched on, the display will show "UP" and \nthe machine  will start in Fast Heat Up mode.  \n4. The heating period depends on the ambient temperature and is approx. 10 minutes. The indicator of the \npump pressure gauge may move slightly during the heating -up phase.  \n5. The Classika PID  is heated up as soon as the desired preset temperature appears on the PID display or \nthe display shows FLU.  While \'FLU\' is shown on the display, the user should perform a flush until \'rdY/Go\' \nis shown on the display. To do this, keep the portafilter clamp ed and place a tall cup under the portafilter \nspout.  \n6. When the message \'rdY/Go\' appears, the machine is ready to brew the first cup of coffee.  \n7. If the user does not perform a flush within the one -minute period (step 4), the display will show the'
Source: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf
=========
FINAL ANSWER: To make a cappuccino with the CLASSIKA PID, first brew the coffee. After brewing the coffee, activate the steam switch by moving it into the upper position to start steam dispensing for frothing the milk. Ensure the steam nozzle is kept underneath the surface of the milk to avoid sprayings and injuries, and do not bring the milk to a boil for successful frothing.
SOURCES: https://www.ecm.de/fileadmin/manual/BA-Classika_PID-2023-08-web.pdf

QUESTION: What should you do if the remote control does not work?
=========
Content: •Remov\ne and reconnect all cables connected to the TV and the external devices. Try new cables if possible.\n •Confirm that \nthe correct input source has been selected (  > Source).\n •Perform a TV\n self diagnosis to determine if the problem is caused by the TV or the device (  >  Settings > \nSupport > Self Diagnosis > Start Picture Test or Start Sound Test).\n •If the \ntest results are normal, reboot the connected devices by unplugging each device's power cord and then \nplugging it in again. If the issue persists, refer to the connection guide in the user manual of the connected device.\n •If y\nou are not using a cable box or satellite box, and your TV is receiving TV signals from an antenna or a cable wall \nconnector, run Auto Program to search for channels (  >  Settings > Broadcasting > Auto Program).\nThe remote control does not work.\n •Check if the power indicator at the bottom of the TV blinks when you press the remote's Power button. If it does \nnot, rep\nlace the remote control's batteries.\n •Make sur\ne that the batteries are installed with their poles (+/–) in the correct direction.\n •Try p\nointing the remote directly at the TV from 5 ft. to 6 ft. ( 1.5~1.8 m) away.\n •If y\nour TV came with a Samsung Smart Remote (Bluetooth Remote), make sure to pair the remote to the TV.\nThe cable box or satellite box remote control doesn’t turn the TV on or off or adjust the volume.\n •Program the cable box or satellite box remote control to operate the TV. Refer to the cable box or satellite box user manual f\nor the SAMSUNG TV code.\nEnglish - 14
Source: https://image-us.samsung.com/SamsungUS/tv-ci-resources/2018-user-manuals/2018_UserManual_Q9FNSeries.pdf

Content:  Pairing the TV to the Samsung Smart Remote\nWhen you turn on the TV for the first time, the Samsung Smart Remote pairs \nto the TV automatically. If the Samsung Smart Remote does not pair to the TV \nautomatically, point it at the remote control sensor of the TV, and then press and hold the buttons labeled \n and  as shown in the figure on the left \nsimultaneously for 3 seconds or more.\nInstalling batteries into the Samsung Smart Remote\n1. Press the  button at the top rear of the Samsung Smart Remote. The body will pop out slightly from the body \ncover.\n2. Turn the r\nemote over, and then slide the body of the remote upwards until the battery compartment is revealed.\n3. If ther\ne are batteries in the remote, press the r  and l buttons on both sides of the battery compartment to \nremove the existing batteries.\n4. Insert tw\no new batteries (1.5V AAA type) into the battery compartment making sure that the batteries' polarities \n(+, -) are oriented correctly. When done, slide the body of the remote until it is nearly in its original position.\n5. Turn the r\nemote over, press and hold the  button on the top rear, and then slide the body of the remote down \ninto place.\n •Alkal\nine batteries are recommended for longer battery life.\nEnglish - 12
Source: https://image-us.samsung.com/SamsungUS/tv-ci-resources/2018-user-manuals/2018_UserManual_Q9FNSeries.pdf

Content: How does Remote Support work?\nYou can easily get Samsung Remote Support service for your TV.\n1. Call the \nSamsung Contact Center and ask for remote support.\n2. Open the menu on \nyour TV, and then go to the Support section. (  >  Settings > Support)\n3. Select R\nemote Management, then read and agree to the service agreements. When the PIN screen appears, \nprovide the PIN number to the technician.\n4. The technician \nwill then access your TV.\nEnglish - 15
Source: https://image-us.samsung.com/SamsungUS/tv-ci-resources/2018-user-manuals/2018_UserManual_Q9FNSeries.pdf
=========
FINAL ANSWER: If your remote control does not work, first check if the power indicator at the bottom of the TV blinks when you press the remote's Power button; if it does not, replace the batteries. Ensure the batteries are installed correctly with their poles (+/–) in the right direction. If using a Samsung Smart Remote, make sure it is paired to the TV, and try pointing the remote directly at the TV from a distance of 5 to 6 feet.
SOURCES:  https://image-us.samsung.com/SamsungUS/tv-ci-resources/2018-user-manuals/2018_UserManual_Q9FNSeries.pdf

QUESTION: How do I clean windshield?
=========
Content:  CLEANING\nIf your windshield is not clear after using the\nwindshield washer or if a wiper blade chatterswhen running, wax or other material may be onthe blade or windshield.\nClean the outside of the windshield with a\nwasher solution or a mild detergent. Your wind-shield is clean if beads do not form whenrinsing with clear water.\nClean each blade by wiping it with a cloth\nsoaked in a washer solution or a mild deter-gent. Then rinse the blade with clear water. Ifyour windshield is still not clear after cleaningthe blades and using the wiper, replace theblades.\nCAUTION\nWorn windshield wiper blades can damage thewindshield and impair driver vision.REPLACING\nReplace the wiper blades if they are worn.\n1. Lift the wiper arm away from the wind-\nshield.\n2. Push the release tab /H17034A, and then move\nthe wiper blade down the wiper arm /H170341\nwhile pushing the release tab to remove.\n3. Insert the new wiper blade onto the wiper\narm until a click sounds.\n4. Rotate the wiper blade so the dimple is in\nthe groove.\nCAUTION\n●After wiper blade replacement, return the\nwiper arm to its original position;\notherwise it may be damaged when the hood\nis opened.\n●Make sure the wiper blades contact the\nglass; otherwise the arm may be damagedfrom wind pressure.\nSDI2048WINDSHIELD WIPER BLADES\n8-18 Maintenance and do-it-yourself'
Source: https://admin.owners.infinitiusa.com/content/manualsandguides/IPL_G_Coupe/2011/2011-Infiniti-IPL-G-Coupe.pdf

Content: ●Do not use an automatic car wash or high\npressure car wash. Avoid applying watercontinuously to the parts around the roofand the windows. This may cause water toleak into the inside of your vehicle.Rinse the vehicle thoroughly with plenty of\nclean water.\nInside flanges, seams and folds on the doors,\nhatches and hood are particularly vulnerable tothe effects of road salt. Therefore, these areasmust be regularly cleaned. Make sure that thedrain holes in the lower edge of the door areopen. Spray water under the body and in thewheel wells to loosen the dirt and wash awayroad salt.\nAvoid leaving water spots on the paint surface\nby using a damp chamois to dry the vehicle.\nWAXING\nRegular waxing protects the paint surface andhelps retain new vehicle appearance. Polishingis recommended to remove built-up wax residueand to avoid a weathered appearance beforereapplying wax.\nAn INFINITI retailer can assist you in choosing\nthe proper product.\n● Wax your vehicle only after a thorough wash-\ning. Follow the instructions supplied with\nthe wax.\n● Do not use a wax containing any abrasives,\ncutting compounds or cleaners that maydamage the vehicle finish.CLEANING EXTERIOR\n7-2 Appearance and care
Source: https://admin.owners.infinitiusa.com/content/manualsandguides/IPL_G_Coupe/2011/2011-Infiniti-IPL-G-Coupe.pdf

Content: Machine compound or aggressive polishing on\na base coat/clear coat paint finish may dull thefinish or leave swirl marks.\nREMOVING SPOTS\nRemove tar and oil spots, industrial dust, in-sects, and tree sap as quickly as possible fromthe paint surface to avoid lasting damage orstaining. Special cleaning products are avail-able at an INFINITI retailer or any automotiveaccessory stores.\nUNDERBODY\nIn areas where road salt is used in winter, theunderbody must be cleaned regularly. This willprevent dirt and salt from building up andcausing the acceleration of corrosion on theunderbody and suspension. Before the winterperiod and again in the spring, the undersealmust be checked and, if necessary, re-treated.\nGLASS\nUse glass cleaner to remove smoke and dustfilm from the glass surfaces. It is normal forglass to become coated with a film after thevehicle is parked in the hot sun. Glass cleanerand a soft cloth will easily remove this film.\nCAUTION\nWhen cleaning the inside of the windows, do\nnot use sharp-edged tools, abrasive cleaners orchlorine-based disinfectant cleaners. Theycould damage the electrical conductors, radioantenna elements or rear window defrosterelements.\nALUMINUM ALLOY WHEELS\nWash regularly with a sponge dampened in amild soap solution, especially during wintermonths in areas where road salt is used. Saltcould discolor the wheels if not removed.\nCAUTION\nFollow the directions below to avoid staining ordiscoloring the wheels:\n●Do not use a cleaner that uses strong acid or\nalkali contents to clean the wheels.\n●Do not apply wheel cleaners to the wheels\nwhen they are hot. The wheel temperature\nshould be the same as ambient temperature.\n●Rinse the wheel to completely remove the\ncleaner within 15 minutes after the cleaner isapplied.CHROME PARTS\nClean chrome parts regularly with a non- abra-sive chrome polish to maintain the finish.\nTIRE DRESSING
Source: https://admin.owners.infinitiusa.com/content/manualsandguides/IPL_G_Coupe/2011/2011-Infiniti-IPL-G-Coupe.pdf
=========
FINAL ANSWER: To clean your windshield, use a washer solution or a mild detergent to clean the outside of the windshield, ensuring that no beads form when rinsing with clear water. Clean each wiper blade by wiping it with a cloth soaked in the same solution or detergent, then rinse the blades with clear water. If the windshield is still not clear after these steps, consider replacing the wiper blades.
SOURCES:  https://admin.owners.infinitiusa.com/content/manualsandguides/IPL_G_Coupe/2011/2011-Infiniti-IPL-G-Coupe.pdf

QUESTION: what can you do
=========
// doesn't matter what the sources are, ignore them
=========
FINAL ANSWER: This is a question-answering system over a corpus of PDF manuals documents defined by the customer.

QUESTION: {question}
=========
{sources}
=========
FINAL ANSWER:"""  # noqa: E501

main = PromptTemplate(template=template, input_variables=["sources", "question"])

per_source = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
