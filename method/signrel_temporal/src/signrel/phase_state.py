"""Fingerprint receipts with dependency invalidation; never infer success from files alone."""
import json
from pathlib import Path
from .utils import fingerprint,sha256,atomic_json

DEPENDENCIES={0:[],1:[0],2:[0,1],3:[0,1],4:[2,3],5:[4],6:[5],7:[5,6],8:[6]}


class PhaseState:
    def __init__(self,path='artifacts/phases.json'):
        self.path=Path(path)
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'schema_version':1,'phases':{}}

    def reusable(self,phase,inputs):
        row=self.state['phases'].get(str(phase),{})
        if row.get('status')!='passed' or row.get('input_fingerprint')!=fingerprint(inputs):return False
        if any(not Path(path).exists() or sha256(path)!=content for path,content in row.get('artifacts',{}).items()):return False
        return all(self.state['phases'].get(str(p),{}).get('status')=='passed' and
                   self.state['phases'][str(p)].get('receipt_fingerprint')==row['dependencies'].get(str(p))
                   for p in DEPENDENCIES[phase])

    def record(self,phase,inputs,artifacts,status,evidence):
        if status not in ('passed','running','incomplete','BLOCKED_EXTERNAL','failed'):raise ValueError('Invalid phase status')
        dependencies={str(p):self.state['phases'].get(str(p),{}).get('receipt_fingerprint') for p in DEPENDENCIES[phase]}
        if status=='passed' and any(self.state['phases'].get(p,{}).get('status')!='passed' for p in dependencies):
            raise ValueError('Phase dependencies have not passed')
        row={'status':status,'input_fingerprint':fingerprint(inputs),'inputs':inputs,'dependencies':dependencies,
             'artifacts':{str(path):sha256(path) for path in artifacts},'evidence':evidence}
        row['receipt_fingerprint']=fingerprint(row)
        previous=self.state['phases'].get(str(phase),{}).get('receipt_fingerprint')
        self.state['phases'][str(phase)]=row
        if previous and previous!=row['receipt_fingerprint']:
            changed={phase}
            for next_phase in sorted(DEPENDENCIES):
                if changed.intersection(DEPENDENCIES[next_phase]) and str(next_phase) in self.state['phases']:
                    self.state['phases'][str(next_phase)]['status']='incomplete';changed.add(next_phase)
        atomic_json(self.path,self.state)
        return row
