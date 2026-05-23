#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "./torch_api.h"

#define SHT_NOBITS 8
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
typedef uint16_t Elf64_Half;
typedef uint32_t Elf64_Word;

#pragma pack(push, 1)
struct Elf64_Ehdr {
  unsigned char e_ident[16];
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;
  Elf64_Half e_shstrndx;
};

struct Elf64_Shdr {
  Elf64_Word sh_name;
  Elf64_Word sh_type;
  uint64_t sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  uint64_t sh_size;
  Elf64_Word sh_link;
  Elf64_Word sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};

struct Elf64_Sym {
  Elf64_Word st_name;
  unsigned char st_info;
  unsigned char st_other;
  Elf64_Half st_shndx;
  Elf64_Addr st_value;
  uint64_t st_size;
};
#pragma pack(pop)

class CubinReader {
private:
  std::map<std::string, uint64_t> symbolOffsets;
  std::string cubinPath;

public:
  CubinReader(const std::string &path) : cubinPath(path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) return;

    Elf64_Ehdr ehdr;
    fs.read((char *)&ehdr, sizeof(ehdr));

    std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
    fs.seekg(ehdr.e_shoff);
    fs.read((char *)shdrs.data(), sizeof(Elf64_Shdr) * ehdr.e_shnum);

    for (int i = 0; i < ehdr.e_shnum; ++i) {
      if (shdrs[i].sh_type == 2) {
        readSymbolTable(fs, shdrs, i);
      }
    }
  }

  uint32_t getUint32(const std::string &name) {
    auto it = symbolOffsets.find(name);
    ASSERT_CHECK(it != symbolOffsets.end(), name);
    if (it->second == 0ULL) return 0;

    std::ifstream fs(cubinPath, std::ios::binary);
    uint32_t val = 0;
    fs.seekg(symbolOffsets[name]);
    fs.read((char *)&val, sizeof(uint32_t));
    return val;
  }

  bool getBool(const std::string &name) {
    return getUint32(name) > 0;
  }

  void readSymbolTable(std::ifstream &fs, const std::vector<Elf64_Shdr> &shdrs, int symIdx) {
    const Elf64_Shdr &symtab = shdrs[symIdx];
    const Elf64_Shdr &strtab = shdrs[symtab.sh_link];

    std::vector<char> names(strtab.sh_size);
    fs.seekg(strtab.sh_offset);
    fs.read(names.data(), strtab.sh_size);

    int count = symtab.sh_size / symtab.sh_entsize;
    std::vector<Elf64_Sym> syms(count);
    fs.seekg(symtab.sh_offset);
    fs.read((char *)syms.data(), symtab.sh_size);

    for (const auto &sym : syms) {
      if (sym.st_name == 0 || sym.st_name >= names.size()) continue;
      if (sym.st_shndx >= shdrs.size()) continue;
      std::string sName = &names[sym.st_name];

      const auto &shdr = shdrs[sym.st_shndx];
      if (shdr.sh_type == SHT_NOBITS) {
        symbolOffsets[sName] = 0;
      } else {
        uint64_t absOffset = shdr.sh_offset + sym.st_value;
        symbolOffsets[sName] = absOffset;
      }
    }
  }
};
