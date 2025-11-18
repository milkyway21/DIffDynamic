"""QVina 对接任务封装，支持从生成分子或原始数据构建任务并解析结果。"""

# 总结：
# - 负责将生成或原始的配体结构与蛋白口袋一起交由 QVina 进行刚体对接。
# - 支持从多种数据源构造任务，并自动处理文件准备、对接执行与结果解析。
# - 可选择是否进行 UFF 优化以及自定义搜索盒中心、尺寸和计算精度。

import os  # 导入操作系统接口。
import subprocess  # 导入子进程管理工具。
import random  # 导入随机数模块。
import string  # 导入字符串常量。
from easydict import EasyDict  # 导入 EasyDict，便于属性访问。
from rdkit import Chem  # 导入 RDKit 化学库。
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule  # 导入 UFF 优化器。

from utils.reconstruct import reconstruct_from_generated  # 导入重建工具，将预测结果转为 RDKit 分子。


def get_random_id(length=30):
    """生成指定长度的随机字符串，用于任务临时文件命名。"""
    letters = string.ascii_lowercase  # 仅使用小写字母。
    return ''.join(random.choice(letters) for i in range(length))  # 逐字符随机拼接。


def load_pdb(path):
    """读取 PDB 文件并返回其文本内容。"""
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path):
    """解析 QVina 输出的 SDF 文件，提取每个姿势的能量与 RMSD。"""
    suppl = Chem.SDMolSupplier(docked_sdf_path)  # 创建 SDF 读取器。
    results = []  # 存储解析结果。
    for i, mol in enumerate(suppl):  # 遍历每个对接姿势。
        if mol is None:  # 过滤损坏的条目。
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]  # 从备注解析能量与 RMSD。
        results.append(EasyDict({
            'rdmol': mol,  # 对接后的 RDKit 分子。
            'mode_id': i,  # 模式编号。
            'affinity': float(line[0]),  # 结合亲和力。
            'rmsd_lb': float(line[1]),  # RMSD 下界。
            'rmsd_ub': float(line[2]),  # RMSD 上界。
        }))

    return results  # 返回所有成功解析的姿势。


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()  # 调用基类构造。
        self.pdb_block = pdb_block  # 保存蛋白结构文本。
        self.ligand_rdmol = ligand_rdmol  # 保存配体分子对象。

    def run(self):
        raise NotImplementedError()  # 需在子类实现运行逻辑。

    def get_results(self):
        raise NotImplementedError()  # 需在子类实现结果获取逻辑。


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        """从生成样本 `ProteinLigandData` 构建对接任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)  # 拼接蛋白路径。
        with open(protein_path, 'r') as f:
            pdb_block = f.read()  # 读取蛋白结构。
        xyz = data.ligand_pos.clone().cpu().tolist()  # 提取配体坐标。
        atomic_nums = data.ligand_element.clone().cpu().tolist()  # 提取配体原子序号。
        ligand_rdmol = reconstruct_from_generated(xyz, atomic_nums)  # 重建 RDKit 配体。
        return cls(pdb_block, ligand_rdmol, **kwargs)  # 返回任务实例。

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        """从生成的 RDKit 分子与其路径构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        """从原始数据集中加载配体与蛋白构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)  # 读取原配体文件。
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))  # 取出 RDKit 分子。
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, conda_env='adt', tmp_dir='./tmp', use_uff=True, center=None,
                 size_factor=1.):
        super().__init__(pdb_block, ligand_rdmol)  # 调用父类构造。
        self.conda_env = conda_env  # 运行 QVina 的 Conda 环境。
        self.tmp_dir = os.path.realpath(tmp_dir)  # 规范化临时目录。
        os.makedirs(tmp_dir, exist_ok=True)  # 确保目录存在。

        self.task_id = get_random_id()  # 生成任务 ID。
        self.receptor_id = self.task_id + '_receptor'  # 受体文件前缀。
        self.ligand_id = self.task_id + '_ligand'  # 配体文件前缀。

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')  # 受体路径。
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')  # 配体路径。

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)  # 写入受体 PDB。

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)  # 加氢并生成坐标。
        if use_uff:
            UFFOptimizeMolecule(ligand_rdmol)  # 可选地执行 UFF 优化。
        sdf_writer = Chem.SDWriter(self.ligand_path)  # 构造 SDF 写入器。
        sdf_writer.write(ligand_rdmol)  # 保存配体。
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol  # 保存优化后的分子。

        pos = ligand_rdmol.GetConformer(0).GetPositions()  # 获取配体坐标。
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2  # 自动居中。
        else:
            self.center = center  # 使用外部指定中心。

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20  # 默认盒子尺寸。
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor  # 基于分子尺寸缩放。

        self.proc = None  # 运行进程句柄。
        self.results = None  # 缓存解析结果。
        self.output = None  # 标准输出缓存。
        self.error_output = None  # 标准错误缓存。
        self.docked_sdf_path = None  # 对接结果路径。

    def run(self, exhaustiveness=16):
        """异步启动 QVina 进程执行对接。"""
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
prepare_receptor4.py -r {receptor_id}.pdb
# Prepare ligand
obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x {size_x} --size_y {size_y} --size_z {size_z} \
    --exhaustiveness {exhaust}
obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
        """.format(
            receptor_id=self.receptor_id,  # 格式化受体前缀。
            ligand_id=self.ligand_id,  # 格式化配体前缀。
            env=self.conda_env,  # 激活的 Conda 环境。
            tmp=self.tmp_dir,  # 临时目录。
            exhaust=exhaustiveness,  # 搜索强度。
            center_x=self.center[0],  # 盒子中心 X。
            center_y=self.center[1],  # 盒子中心 Y。
            center_z=self.center[2],  # 盒子中心 Z。
            size_x=self.size_x,  # 盒子尺寸 X。
            size_y=self.size_y,  # 盒子尺寸 Y。
            size_z=self.size_z  # 盒子尺寸 Z。
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)  # 记录输出 SDF 路径。

        self.proc = subprocess.Popen(  # 启动 Bash 子进程。
            '/bin/bash',
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))  # 将命令写入子进程。
        self.proc.stdin.close()  # 关闭标准输入，触发执行。

        # return commands

    def run_sync(self):
        """同步执行对接并返回结果。"""
        self.run()  # 启动异步流程。
        while self.get_results() is None:  # 轮询直至完成。
            pass
        results = self.get_results()  # 获取最终结果。
        print('Best affinity:', results[0]['affinity'])  # 输出最佳亲和力。
        return results  # 返回姿势列表。

    def get_results(self):
        """检查对接进程状态并解析结果。"""
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:  # 仅首次解析。
                self.output = self.proc.stdout.readlines()  # 缓存标准输出。
                self.error_output = self.proc.stderr.readlines()  # 缓存错误输出。
                try:
                    self.results = parse_qvina_outputs(self.docked_sdf_path)  # 解析 SDF。
                except:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)  # 打印异常信息。
                    return []
            return self.results  # 返回缓存结果。
