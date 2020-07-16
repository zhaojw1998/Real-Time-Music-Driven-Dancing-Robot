function sysCall_init() 
    modelBase=sim.getObjectAssociatedWithScript(sim.handle_self)
    ref=sim.getObjectHandle('GyroSensor_reference')
    ui=simGetUIHandle('GyroSensor_UI')
    simSetUIButtonLabel(ui,0,sim.getObjectName(modelBase))
    gyroCommunicationTube=sim.tubeOpen(0,'gyroData'..sim.getNameSuffix(nil),1)
    oldTransformationMatrix=sim.getObjectMatrix(ref,-1)
    lastTime=sim.getSimulationTime()
    simRemoteApi.start(19999)
end

function sysCall_cleanup() 
 
end 

function sysCall_sensing() 
    local transformationMatrix=sim.getObjectMatrix(ref,-1)
    local oldInverse=simGetInvertedMatrix(oldTransformationMatrix)
    local m=sim.multiplyMatrices(oldInverse,transformationMatrix)
    local euler=sim.getEulerAnglesFromMatrix(m)
    local currentTime=sim.getSimulationTime()
    local gyroData={0,0,0}
    local dt=currentTime-lastTime
    if (dt~=0) then
        gyroData[1]=euler[1]/dt
        gyroData[2]=euler[2]/dt
        gyroData[3]=euler[3]/dt
    end
    sim.tubeWrite(gyroCommunicationTube,sim.packFloatTable(gyroData))
    simSetUIButtonLabel(ui,3,string.format("X-Gyro: %.4f",gyroData[1]))
    simSetUIButtonLabel(ui,4,string.format("Y-Gyro: %.4f",gyroData[2]))
    simSetUIButtonLabel(ui,5,string.format("Z-Gyro: %.4f",gyroData[3]))
    gyro = sim.packFloatTable(gyroData)
    sim.setStringSignal('Gyrometer', gyro)
    oldTransformationMatrix=sim.copyMatrix(transformationMatrix)
    lastTime=currentTime

end 
