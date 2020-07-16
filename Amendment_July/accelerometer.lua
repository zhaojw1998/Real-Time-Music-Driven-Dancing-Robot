function sysCall_init() 
    modelBase=sim.getObjectAssociatedWithScript(sim.handle_self)
    massObject=sim.getObjectHandle('Accelerometer_mass')
    sensor=sim.getObjectHandle('Accelerometer_forceSensor')
    result,mass=sim.getObjectFloatParameter(massObject,sim.shapefloatparam_mass)
    ui=simGetUIHandle('Accelerometer_UI')
    simSetUIButtonLabel(ui,0,sim.getObjectName(modelBase))
    accelCommunicationTube=sim.tubeOpen(0,'accelerometerData'..sim.getNameSuffix(nil),1)
    simRemoteApi.start(19999)
end
-- Check the end of the script for some explanations!

function sysCall_cleanup() 
 
end 

function sysCall_sensing() 
    result,force=sim.readForceSensor(sensor)
    if (result>0) then
        accel={force[1]/mass,force[2]/mass,force[3]/mass}
        sim.tubeWrite(accelCommunicationTube,sim.packFloatTable(accel))
        simSetUIButtonLabel(ui,3,string.format("X-Accel: %.4f",accel[1]))
        simSetUIButtonLabel(ui,4,string.format("Y-Accel: %.4f",accel[2]))
        simSetUIButtonLabel(ui,5,string.format("Z-Accel: %.4f",accel[3]))
        acc = sim.packFloatTable(accel)
        sim.setStringSignal('Acceleration', acc)
    else
        acc = sim.packFloatTable(0)
        sim.setStringSignal('Acceleration', acc)
        simSetUIButtonLabel(ui,3,"X-Accel: -")
        simSetUIButtonLabel(ui,4,"Y-Accel: -")
        simSetUIButtonLabel(ui,5,"Z-Accel: -")
    end
    
end 